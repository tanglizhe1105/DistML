package org.apache.spark.mllib.topicModeling

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, _}
import breeze.numerics.{abs, digamma, exp, _}
import com.intel.distml.api.{Model, Session}
import com.intel.distml.util.{IntArrayWithIntKey, KeyList}
import com.intel.distml.util.scala.{DoubleArrayWithIntKey, DoubleMatrixWithIntKey}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

class SuffStats(
                 val T: Int,
                 val Wt: Int,
                 val m_chunksize: Int) extends Serializable {
  var m_var_sticks_ss = BDV.zeros[Double](T)
  var m_var_beta_ss = BDM.zeros[Double](T, Wt)

  def set_zero: Unit = {
    m_var_sticks_ss = BDV.zeros(T)
    m_var_beta_ss = BDM.zeros(T, Wt)
  }
}

object OnlineHDPOptimizer extends Serializable {
  val rhot_bound = 0.0

  def log_normalize(v: BDV[Double]): (BDV[Double], Double) = {
    val log_max = 100.0
    val max_val = v.toArray.max
    val log_shift = log_max - log(v.size + 1.0) - max_val
    val tot: Double = sum(exp(v + log_shift))
    val log_norm = log(tot) - log_shift
    (v - log_norm, log_norm)
  }

  def log_normalize(m: BDM[Double]): (BDM[Double], BDV[Double]) = {
    val log_max = 100.0
    // get max for every row
    val max_val: BDV[Double] = m(*, ::).map(v => max(v))
    val log_shift: BDV[Double] = log_max - log(m.cols + 1.0) - max_val

    val m_shift: BDM[Double] = exp(m(::, *) + log_shift)
    val tot: BDV[Double] = sum(m_shift(*, ::))

    val log_norm: BDV[Double] = log(tot) - log_shift
    (m(::, *) - log_norm, log_norm)
  }

  def expect_log_sticks(m: BDM[Double]): BDV[Double] = {
    //    """
    //    For stick-breaking hdp, return the E[log(sticks)]
    //    """
    val column = sum(m(::, *))

    val dig_sum: BDV[Double] = digamma(column.toDenseVector)
    val ElogW: BDV[Double] = digamma(m(0, ::).inner) - dig_sum
    val Elog1_W: BDV[Double] = digamma(m(1, ::).inner) - dig_sum
    //
    val n = m.cols + 1
    val Elogsticks = BDV.zeros[Double](n)
    Elogsticks(0 until n - 1) := ElogW(0 until n - 1)
    val cs = accumulate(Elog1_W)

    Elogsticks(1 until n) := Elogsticks(1 until n) + cs
    Elogsticks
  }

  /**
    * For theta ~ Dir(alpha), computes E[log(theta)] given alpha. Currently the implementation
    * uses digamma which is accurate but expensive.
    */
  private def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

}

/**
  * Implemented based on the paper "Online Variational Inference for the Hierarchical Dirichlet Process" (Chong Wang, John Paisley and David M. Blei)
  */

class OnlineHDPOptimizer(
                          var m_D: Long = 0,
                          val m_windowSize: Int = 256,
                          val m_W: Int = 0,
                          val m_K: Int = 15,
                          val m_T: Int = 150,
                          val m_kappa: Double = 1.0,
                          var m_tau: Double = 64.0,
                          val m_alpha: Double = 1,
                          val m_gamma: Double = 1,
                          val m_eta: Double = 0.01,
                          val m_scale: Double = 1.0,
                          val m_var_converge: Double = 0.0001
                        ) extends Serializable {


  val lda_alpha: Double = 0.1D
  val lda_beta: Double = 0.01D
  var rhot = 0.0
  val rhot_bound = 0.0
  var m_iteration = 0
  var m_status_up_to_date = true

  var m_lambda: BDM[Double] = null
  val m_timestamp: BDV[Int] = BDV.zeros[Int](m_W)
  val m_r = collection.mutable.MutableList[Double](0)
  /*  val m_var_sticks = BDM.zeros[Double](2, m_T - 1) // 2 * T - 1
    m_var_sticks(0, ::) := 1.0
    m_var_sticks(1, ::) := new BDV[Double]((m_T - 1 to 1 by -1).map(_.toDouble).toArray).t
    // T * W
    val m_lambda: BDM[Double] = BDM.rand(m_T, m_W) :* ((m_D.toDouble) * 100.0 / (m_T * m_W).toDouble) - m_eta
    var m_lambda_sum = sum(m_lambda(*, ::)) // row sum*/

  // T * W
  //val m_Elogbeta = OnlineHDPOptimizer.dirichletExpectation(m_lambda + m_eta)

  /*  val m_timestamp: BDV[Int] = BDV.zeros[Int](m_W)
    val m_r = collection.mutable.MutableList[Double](0)*/

  def initPSModel(m: Model, monitorPath: String) = {
    // init ps
    var start = System.currentTimeMillis()
    var end = start
    println("[ps model Init start: ")
    // init lambda
    m_lambda = BDM.rand(m_T, m_W) :* ((m_D.toDouble) * 100.0 / (m_T * m_W).toDouble) - m_eta
    // init lambda sum
    val lambdaSum = sum(m_lambda(*, ::))

    val session = new Session(m, monitorPath, 0)

    val lambdaSumModel = m.getMatrix("lambda_sum").asInstanceOf[DoubleArrayWithIntKey]
    val wtSum = new mutable.HashMap[Int, Double]
    for (i <- 0 until m_T) {
      wtSum.put(i, lambdaSum(i))
    }
    lambdaSumModel.push(wtSum, session)

    // init global sticks
    /*    val global_sticks = BDM.zeros[Double](2, m_T - 1) // 2 * T - 1
        global_sticks(0, ::) := 1.0
        global_sticks(1, ::) := new BDV[Double]((m_T - 1 to 1 by -1).map(_.toDouble).toArray).t*/
    /*    val varPhiModel = m.getMatrix("var_phi").asInstanceOf[DoubleArrayWithIntKey]
        val varPhi = new mutable.HashMap[Int, Double] // asume var phi is doc topic count
        for (i <- 0 until m_T) {
          val temp = 0D
          varPhi.put(i, temp)
        }
        varPhiModel.push(varPhi, session)*/

    val lambdaModel = m.getMatrix("lambda").asInstanceOf[DoubleMatrixWithIntKey]
    val wt = new mutable.HashMap[Int, Array[Double]]
    for (i <- 0 until m_W) {
      val topicArray = m_lambda(::, i).toArray
      wt.put(i, topicArray)
      if (i > 0 && i % 1000 == 0) {
        lambdaModel.push(wt, session)
        wt.clear()
      }
    }
    if (wt.nonEmpty) {
      lambdaModel.push(wt, session)
    }

    session.disconnect()
    end = System.currentTimeMillis()
    println(s"[model init push , partition 0 time ${end - start} ms]")
    start = end
  }

  def next(m: Model, monitorPath: String)(batch: RDD[(Long, Vector)]): Double = {
    rhot = m_scale * pow(m_tau + m_iteration, -m_kappa)
    if (rhot < rhot_bound)
      rhot = rhot_bound

    val (docScore, vocabs) = batch.mapPartitionsWithIndex(inference(m, monitorPath)).reduce((x, y) => (x._1+y._1, x._2++y._2))

    m_iteration += 1
    m_timestamp(vocabs.toList) := m_iteration
    m_r += (m_r.last + log(1 - rhot))

    docScore
  }

  def inference(m: Model, monitorPath: String)(index: Int, it: Iterator[(Long, Vector)]): Iterator[(Double, mutable.HashSet[Int])] = {
    // solve documents for each partition parrerrally
    val docs = it.toArray
    // compute local words in this partition
    val local2globle = new mutable.HashMap[Int, Int] // local word id to globel word id
    val globle2local = new mutable.HashMap[Int, Int] // globel word id to local word id
    val vocabset = new mutable.HashSet[Int]
    docs.foreach { doc =>
      val termCounts = doc._2
      val (ids: List[Int], _) = termCounts match {
        case v: DenseVector => ((0 until v.size).toList, v.values)
        case v: SparseVector => (v.indices.toList, v.values)
        case v => throw new IllegalArgumentException("Online LDA does not support vector type "
          + v.getClass)
      }
      vocabset ++= ids
    }

    var start = System.currentTimeMillis()
    var end = start
    //fetch globle parameters
    val session = new Session(m, monitorPath, index)


    val topicIds = new KeyList()
    for (id <- 0 until m_T) {
      topicIds.addKey(id)
    }
    // get word topic sum from ps in this partition
    val lambdaSumModel = m.getMatrix("lambda_sum").asInstanceOf[DoubleArrayWithIntKey]
    val lambdaSum: mutable.HashMap[Int, Double] = lambdaSumModel.fetch(topicIds, session)
    val lambda_sum = BDV.zeros[Double](topicIds.size().toInt)
    for (id <- 0 until topicIds.size().toInt) {
      lambdaSum.get(id) match {
        case Some(sum) => lambda_sum(id) = sum
        case None => throw new IllegalArgumentException("Convert ps to lambda sum ")
      }
    }
    // get var phi
    val varPhiModel = m.getMatrix("var_phi").asInstanceOf[DoubleArrayWithIntKey]
    val varPhi: mutable.HashMap[Int, Double] = varPhiModel.fetch(topicIds, session)
    //convert ps word topic to matrix
    val m_var_phi = BDV.zeros[Double](topicIds.size().toInt) // var phi
    for (id <- 0 until topicIds.size().toInt) {
      val vpf = varPhi.get(id) match {
        case Some(f) => f
        case None => throw new IllegalArgumentException("Convert ps to global sticks ")
      }
      m_var_phi(id) = vpf
    }

    val word_list = vocabset.toList.sorted
    val vocabs = new KeyList()
    for (id <- word_list) {
      globle2local += id -> vocabs.size().toInt
      local2globle += vocabs.size().toInt -> id
      vocabs.addKey(id)
    }
/*    // get timestamp
    val timestampModel = m.getMatrix("timestamp").asInstanceOf[IntArrayWithIntKey]
    val timekv: java.util.HashMap[Integer, Integer] = timestampModel.fetch(vocabs, session)
    val timestamp = BDV.zeros[Int](word_list.length)
    for (i <- word_list.indices) {
      timestamp(i) = timekv.get(local2globle(i))
    }*/
    // get word topic from ps in this partition
    val lambdaModel = m.getMatrix("lambda").asInstanceOf[DoubleMatrixWithIntKey]
    val wt: mutable.HashMap[Int, Array[Double]] = lambdaModel.fetch(vocabs, session)
    //convert ps word topic to matrix
    val lambda = BDM.zeros[Double](m_T, word_list.length) // local Elogbeta
    for (i <- word_list.indices) {
      val topic = wt.get(local2globle(i)) match {
        case Some(topicArray) => BDV(topicArray)
        case None => throw new IllegalArgumentException("Convert ps to lambda ")
      }
      lambda(::, i) := topic
    }
    val rw: BDV[Double] = new BDV(word_list.indices.map(id => m_timestamp(id)).map(t => m_r(t)).toArray)
    val exprw: BDV[Double] = exp(rw.map(d => m_r.last - d))
    val wordsMatrix = lambda.copy
    for (row <- 0 until wordsMatrix.rows) {
      wordsMatrix(row, ::) := (wordsMatrix(row, ::).t :* exprw).t
    }
    val Elogbeta = wordsMatrix.copy // local Elogbeta
    for (i <- word_list.indices) {
      Elogbeta(::, i) := digamma(wordsMatrix(::, i) + m_eta) - digamma(lambda_sum + m_W * m_eta)
    }
    end = System.currentTimeMillis()
    println(s"[model train fetch , partition $index time ${end - start} ms]")
    start = end

    // compute var sticks
    val m_var_sticks = BDM.zeros[Double](2, topicIds.size().toInt - 1) // var phi
    if(m_iteration == 0){
      m_var_sticks(0, ::) := 1.0
      m_var_sticks(1, ::) := new BDV[Double]((m_T - 1 to 1 by -1).map(_.toDouble).toArray).t
    }else{
      m_var_sticks(0, ::) := (m_var_phi(0 to m_T - 2) + 1.0).t
      val var_phi_sum = flipud(m_var_phi(1 until m_var_phi.length)) // T - 1
      m_var_sticks(1, ::) := (flipud(accumulate(var_phi_sum)) + m_gamma).t
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // run variational inference on some new docs
    var score = 0.0
    var count = 0D
    val dict = globle2local.toMap
    val ss = new SuffStats(m_T, word_list.length, docs.length)
    val Elogsticks_1st: BDV[Double] = OnlineHDPOptimizer.expect_log_sticks(m_var_sticks) // global sticks
    docs.foreach(doc =>
      if (doc._2.size > 0) {
        val doc_word_ids = doc._2.asInstanceOf[SparseVector].indices
        val doc_word_counts = doc._2.asInstanceOf[SparseVector].values
        val wl = doc_word_ids.toList

        val doc_score = doc_e_step(doc, ss, Elogbeta, Elogsticks_1st,
          word_list, dict, wl, new BDV[Double](doc_word_counts), m_var_converge)
        count += sum(doc_word_counts)
        score += doc_score
      }
    )
    end = System.currentTimeMillis()
    println(s"[model train train , partition $index time ${end - start} ms]")
    start = end

    // update global parameters
    // compute new lambda and delta lambda
    val lambda_new = ss.m_var_beta_ss * (m_D.toDouble / m_windowSize.toDouble) // partition new lamda
    lambda :*= (docs.length.toDouble / m_windowSize.toDouble) // partition portition
    lambda_new :-= lambda // delta lambda
    lambda_new :*= rhot
    // push delta lambda
    for (i <- word_list.indices) {
      wt(local2globle(i)) = lambda_new(::, i).toArray
    }
    lambdaModel.push(wt, session)
//    // push timestamp
//    val timestampNew = timestamp.copy
//    timestampNew := m_iteration +1
//    timestampNew :-= timestamp
//    for (i <- word_list.indices) {
//      timekv.put(local2globle(i), timestampNew(i))
//    }
//    timestampModel.push(timekv, session)

    // compute & push lambda sum
    lambda_sum := sum(lambda_new(*, ::)) // delta lambda sum
    for (id <- 0 until topicIds.size().toInt) {
      lambdaSum(id) = lambda_sum(id)
    }
    lambdaSumModel.push(lambdaSum, session)

    // compute & push var phi
    val varphi_new = ss.m_var_sticks_ss * (m_D.toDouble / m_windowSize.toDouble) // partition new var phi
    m_var_phi :*= (docs.length.toDouble / m_windowSize.toDouble) // partition portition
    varphi_new :-= m_var_phi
    varphi_new :*= rhot
    for (id <- 0 until topicIds.size().toInt) {
      varPhi(id) = varphi_new(id)
    }
    varPhiModel.push(varPhi, session)
    end = System.currentTimeMillis()
    println(s"[model train push , partition $index time ${end - start} ms]")


    session.progress(docs.length)
    session.disconnect()

    Iterator((score, vocabset))
  }


  def doc_e_step(doc: (Long, Vector),
                 ss: SuffStats,
                 m_Elogbeta: BDM[Double],
                 Elogsticks_1st: BDV[Double],
                 word_list: List[Int],
                 dict: Map[Int, Int], // global to partition local
                 doc_word_ids: List[Int],
                 doc_word_counts: BDV[Double],
                 var_converge: Double): Double = {

    val chunkids = doc_word_ids.map(id => dict(id)) // partition local

    val Elogbeta_doc: BDM[Double] = m_Elogbeta(::, chunkids).toDenseMatrix // T * Wt
    // very similar to the hdp equations, 2 * K - 1
    val v = BDM.zeros[Double](2, m_K - 1)
    v(0, ::) := 1.0
    v(1, ::) := m_alpha

    var Elogsticks_2nd = OnlineHDPOptimizer.expect_log_sticks(v)

    // back to the uniform
    var phi: BDM[Double] = BDM.ones[Double](doc_word_ids.size, m_K) * 1.0 / m_K.toDouble // Wt * K

    var likelihood = 0.0
    var old_likelihood = -1e200
    var converge = 1.0
    val eps = 1e-100

    var iter = 0
    val max_iter = 100

    var var_phi_out: BDM[Double] = null

    // not yet support second level optimization yet, to be done in the future
    while (iter < max_iter && converge > var_converge) {

      // var_phi
      val (log_var_phi: BDM[Double], var_phi: BDM[Double]) =
        if (iter < 3) {
          val element = Elogbeta_doc.copy // T * Wt
          for (i <- 0 until element.rows) {
            element(i, ::) := (element(i, ::).t :* doc_word_counts).t
          }
          var var_phi: BDM[Double] = phi.t * element.t // K * Wt   *  Wt * T  => K * T
          val (log_var_phi, log_norm) = OnlineHDPOptimizer.log_normalize(var_phi)
          var_phi = exp(log_var_phi)
          (log_var_phi, var_phi)
        }
        else {
          val element = Elogbeta_doc.copy
          for (i <- 0 until element.rows) {
            element(i, ::) := (element(i, ::).t :* doc_word_counts).t
          }
          val product: BDM[Double] = phi.t * element.t
          for (i <- 0 until product.rows) {
            product(i, ::) := (product(i, ::).t + Elogsticks_1st).t
          }

          var var_phi: BDM[Double] = product
          val (log_var_phi, log_norm) = OnlineHDPOptimizer.log_normalize(var_phi)
          var_phi = exp(log_var_phi)
          (log_var_phi, var_phi)
        }

      val (log_phi, log_norm) =
      // phi
        if (iter < 3) {
          phi = (var_phi * Elogbeta_doc).t
          val (log_phi, log_norm) = OnlineHDPOptimizer.log_normalize(phi)
          phi = exp(log_phi)
          (log_phi, log_norm)
        }
        else {
          //     K * T       T * Wt
          val product: BDM[Double] = (var_phi * Elogbeta_doc).t
          for (i <- 0 until product.rows) {
            product(i, ::) := (product(i, ::).t + Elogsticks_2nd).t
          }
          phi = product
          val (log_phi, log_norm) = OnlineHDPOptimizer.log_normalize(phi)
          phi = exp(log_phi)
          (log_phi, log_norm)
        }


      // v
      val phi_all = phi.copy
      for (i <- 0 until phi_all.cols) {
        phi_all(::, i) := phi_all(::, i) :* doc_word_counts
      }

      v(0, ::) := sum(phi_all(::, m_K - 1)) + 1.0
      val selected = phi_all(::, 1 until m_K)
      val t_sum = sum(selected(::, *)).toDenseVector
      val phi_cum = flipud(t_sum)
      v(1, ::) := (flipud(accumulate(phi_cum)) + m_alpha).t
      Elogsticks_2nd = OnlineHDPOptimizer.expect_log_sticks(v)

      likelihood = 0.0
      // compute likelihood
      // var_phi part/ C in john's notation

      val diff = log_var_phi.copy
      for (i <- 0 until diff.rows) {
        diff(i, ::) := (Elogsticks_1st :- diff(i, ::).t).t
      }

      likelihood += sum(diff :* var_phi)

      // v part/ v in john's notation, john's beta is alpha here
      val log_alpha = log(m_alpha)
      likelihood += (m_K - 1) * log_alpha
      val dig_sum = digamma(sum(v(::, *))).toDenseVector
      val vCopy = v.copy
      for (i <- 0 until v.cols) {
        vCopy(::, i) := BDV[Double](1.0, m_alpha) - vCopy(::, i)
      }

      val dv = digamma(v)
      for (i <- 0 until v.rows) {
        dv(i, ::) := dv(i, ::) - dig_sum.t
      }

      likelihood += sum(vCopy :* dv)
      likelihood -= sum(lgamma(sum(v(::, *)))) - sum(lgamma(v))

      // Z part
      val log_phiCopy = log_phi.copy
      for (i <- 0 until log_phiCopy.rows) {
        log_phiCopy(i, ::) := (Elogsticks_2nd - log_phiCopy(i, ::).t).t
      }
      likelihood += sum(log_phiCopy :* phi)

      // X part, the data part
      val Elogbeta_docCopy = Elogbeta_doc.copy
      for (i <- 0 until Elogbeta_docCopy.rows) {
        Elogbeta_docCopy(i, ::) := (Elogbeta_docCopy(i, ::).t :* doc_word_counts).t
      }

      likelihood += sum(phi.t :* (var_phi * Elogbeta_docCopy))

      converge = (likelihood - old_likelihood) / abs(old_likelihood)
      old_likelihood = likelihood

      if (converge < -0.000001)
        println("likelihood is decreasing!")

      iter += 1
      var_phi_out = var_phi
    }

    // update the suff_stat ss
    // this time it only contains information from one doc
    val sumPhiOut = sum(var_phi_out(::, *))
    ss.m_var_sticks_ss += sumPhiOut.toDenseVector

    /*    val phiCopy = phi.copy.t
        for (i <- 0 until phi.rows) {
          phiCopy(i, ::) := (phiCopy(i, ::).t :* doc_word_counts).t
        }*/
    for (i <- 0 until phi.cols) {
      phi(::, i) :*= doc_word_counts
    }

    val middleResult: BDM[Double] = var_phi_out.t * phi.t // T K * K * W => T * W
    for (i <- chunkids.indices) {
      ss.m_var_beta_ss(::, chunkids(i)) :+= middleResult(::, i)
    }

    likelihood
  }

  def topicPerplexity(m: Model, monitorPath: String): Double = {
    var start = System.currentTimeMillis()
    var end = start
    // get word topic from ps in this partition
    val vocabs = new KeyList()
    for (id <- 0 until m_W) {
      vocabs.addKey(id)
    }
    val session = new Session(m, monitorPath, 0)
    val wtm = m.getMatrix("lambda").asInstanceOf[DoubleMatrixWithIntKey]
    val wt: mutable.HashMap[Int, Array[Double]] = wtm.fetch(vocabs, session)
    session.disconnect()
    end = System.currentTimeMillis()
    println(s"[model train fetch , partition 0 time ${end - start} ms]")
    start = end

    for (i <- 0 until m_W) {
      val topic = wt.get(i) match {
        case Some(topicArray) => BDV(topicArray)
        case None => throw new IllegalArgumentException("Convert ps word topic to matrix ")
      }
      m_lambda(::, i) := topic // fill local lambda with ps parameter
    }

    val m_Elogbeta = OnlineHDPOptimizer.dirichletExpectation(m_lambda + m_eta)
    // E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
    var topicScore = 0D
    val sumEta = m_eta * m_W
    topicScore += sum((m_eta - m_lambda) :* m_Elogbeta)
    topicScore += sum(lgamma(m_lambda) - lgamma(m_eta))
    topicScore += sum(lgamma(sumEta) - lgamma(sum(m_lambda(::, breeze.linalg.*))))

    topicScore
  }
}