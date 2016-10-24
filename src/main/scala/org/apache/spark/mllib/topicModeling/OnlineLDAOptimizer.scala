package org.apache.spark.mllib.topicModeling

import java.util.Random

import breeze.linalg.{max, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import breeze.stats.distributions.{Gamma, RandBasis}
import com.intel.distml.api.{Model, Session}
import com.intel.distml.platform.DistML
import com.intel.distml.util.scala.FloatMatrixWithIntKey
import com.intel.distml.util.{DataStore, KeyList}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * :: Experimental ::
  *
  * An Optimizer based on OnlineLDAOptimizer, that can also output the the document~topic
  * distribution
  *
  * An early version of the implementation was merged into MLlib (PR #4419), and several extensions (e.g., predict) are added here
  *
  */
object OnlineLDAOptimizer {
  /**
    * For theta ~ Dir(alpha), computes E[log(theta)] given alpha. Currently the implementation
    * uses digamma which is accurate but expensive.
    */
  def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }
}

final class OnlineLDAOptimizer extends LDAOptimizer with Serializable {
  // LDA common parameters
  private var k: Int = 0
  private var corpusSize: Long = 0
  private var windowSize: Int = 0
  // model equally document size
  private var partitionSize: Int = 0
  private var vocabSize: Int = 0

  /** alias for docConcentration */
  private var alpha: Double = 0

  /** (private[lda] for debugging)  Get docConcentration */
  private[topicModeling] def getAlpha: Double = alpha

  /** alias for topicConcentration */
  private var eta: Double = 0

  /** (private[lda] for debugging)  Get topicConcentration */
  private[topicModeling] def getEta: Double = eta

  private var randomGenerator: java.util.Random = null

  // Online LDA specific parameters
  // Learning rate is: (tau0 + t)^{-kappa}
  private var tau0: Double = 1024
  private var kappa: Double = 0.51
  private var miniBatchFraction: Double = 0.05

  // internal data structure
  private var docs: RDD[(Long, Vector)] = null

  /** Dirichlet parameter for the posterior over topics */
  private var lambda: BDM[Double] = null

  /** (private[lda] for debugging) Get parameter for topics */
  private[topicModeling] def getLambda: BDM[Double] = lambda

  /** Current iteration (count of invocations of [[next()]]) */
  private var iteration: Int = 0
  private var gammaShape: Double = 100

  /**
    * A (positive) learning parameter that downweights early iterations. Larger values make early
    * iterations count less.
    */
  def getTau0: Double = this.tau0

  /**
    * A (positive) learning parameter that downweights early iterations. Larger values make early
    * iterations count less.
    * Default: 1024, following the original Online LDA paper.
    */
  def setTau0(tau0: Double): this.type = {
    require(tau0 > 0, s"LDA tau0 must be positive, but was set to $tau0")
    this.tau0 = tau0
    this
  }

  /**
    * Learning rate: exponential decay rate
    */
  def getKappa: Double = this.kappa

  /**
    * Learning rate: exponential decay rate---should be between
    * (0.5, 1.0] to guarantee asymptotic convergence.
    * Default: 0.51, based on the original Online LDA paper.
    */
  def setKappa(kappa: Double): this.type = {
    require(kappa >= 0, s"Online LDA kappa must be nonnegative, but was set to $kappa")
    this.kappa = kappa
    this
  }

  /**
    * Mini-batch fraction, which sets the fraction of document sampled and used in each iteration
    */
  def getMiniBatchFraction: Double = this.miniBatchFraction

  /**
    * Mini-batch fraction in (0, 1], which sets the fraction of document sampled and used in
    * each iteration.
    *
    * Note that this should be adjusted in synch with
    * so the entire corpus is used.  Specifically, set both so that
    * maxIterations * miniBatchFraction >= 1.
    *
    * Default: 0.05, i.e., 5% of total documents.
    */
  def setMiniBatchFraction(miniBatchFraction: Double): this.type = {
    require(miniBatchFraction > 0.0 && miniBatchFraction <= 1.0,
      s"Online LDA miniBatchFraction must be in range (0,1], but was set to $miniBatchFraction")
    this.miniBatchFraction = miniBatchFraction
    this
  }

  /**
    * (private[lda])
    * Set the Dirichlet parameter for the posterior over topics.
    * This is only used for testing now. In the future, it can help support training stop/resume.
    */
  private[topicModeling] def setLambda(lambda: BDM[Double]): this.type = {
    this.lambda = lambda
    this
  }

  /**
    * (private[lda])
    * Used for random initialization of the variational parameters.
    * Larger value produces values closer to 1.0.
    * This is only used for testing currently.
    */
  private[topicModeling] def setGammaShape(shape: Double): this.type = {
    this.gammaShape = shape
    this
  }

  override def initialize(sc: SparkContext, docs: RDD[(Long, Vector)], lda: LDA): this.type = {
    null
  }

  def initialize(m : Model, monitorPath : String)(sc: SparkContext, lda: LDA): this.type = {
    this.corpusSize = lda.getCorpusSize
    this.windowSize = lda.getWindowSize
    this.partitionSize = lda.getPartition
    this.vocabSize = lda.getVocabSize
    this.k = lda.getK
    this.alpha = if (lda.getDocConcentration == -1) 1.0 / k else lda.getDocConcentration
    this.eta = if (lda.getTopicConcentration == -1) 1.0 / k else lda.getTopicConcentration
    this.randomGenerator = new Random(lda.getSeed)

    this.docs = null
    this.lambda = BDM.zeros[Double](k, vocabSize)

    // init ps
    var start = System.currentTimeMillis()
    var end = start
    println("[ps model Init start: ")
    val session = new Session(m, monitorPath, 0)
    val wtm = m.getMatrix("word-topics").asInstanceOf[FloatMatrixWithIntKey]
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val wt = new mutable.HashMap[Int, Array[Float]]
    for (i <- 0 until vocabSize) {
      val temp = gammaRandomGenerator.sample(k).toArray.map(_.toFloat)
      wt.put(i, temp)
      if (i > 0 && i % 1000 == 0) {
        wtm.push(wt, session)
        wt.clear()
      }
    }
    if (wt nonEmpty) {
      wtm.push(wt, session)
    }
    session.disconnect()
    end = System.currentTimeMillis()
    println(s"[model init push , partition 0 time ${end - start} ms]")
    start = end

    this.iteration = 0
    this
  }

  override def next(): this.type = {
    null
  }

  def next(m : Model, monitorPath : String)(batch: RDD[(Long, Vector)]): RDD[(Long, BDV[Double])] = {
    if (batch.isEmpty()) return null
    iteration += 1

    batch.mapPartitionsWithIndex(inference(m, monitorPath))
  }

  def inference(m : Model, monitorPath : String)(index: Int, it: Iterator[(Long, Vector)]): Iterator[(Long, BDV[Double])] = {
    // solve documents for each partition parrerrally
    val docs = it.toArray
    // compute local words in this partition
    val local2globle = new mutable.HashMap[Int, Int]  // local word id to globel word id
    val globle2local = new mutable.HashMap[Int, Int]  // globel word id to local word id
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
    val vocabs = new KeyList()
    for (id <- vocabset.toArray.sorted) {
      globle2local += id -> vocabs.size().toInt
      local2globle += vocabs.size().toInt -> id
      vocabs.addKey(id)
    }

    var start = System.currentTimeMillis()
    var end = start
    // get word topic from ps in this partition
    val session = new Session(m, monitorPath, index)
    val wtm = m.getMatrix("word-topics").asInstanceOf[FloatMatrixWithIntKey]
    val wt: mutable.HashMap[Int, Array[Float]] = wtm.fetch(vocabs, session)
    end = System.currentTimeMillis()
    println(s"[model train fetch , partition $index time ${end - start} ms]")
    start = end

    //convert ps word topic to matrix
    val lambda = BDM.zeros[Double](k, vocabs.size().toInt)  // local lambda
    for (i <- 0 until vocabs.size().toInt) {
      val topic = wt.get(local2globle(i)) match {
        case Some(topicArray) => topicArray
        case None => throw new IllegalArgumentException("Convert ps word topic to matrix ")
      }
      lambda(::, i) := BDV(topic.map(_.toDouble))  // fill local lambda with ps parameter
    }

    // variational bayes inference
    val Elogbeta = OnlineLDAOptimizer.dirichletExpectation(lambda)  // K * ids
    val expElogbeta = exp(Elogbeta)  // K * ids
    val stat = BDM.zeros[Double](k, vocabs.size().toInt)
    val gammaList = new mutable.ArrayBuffer[(Long, BDV[Double])]

    docs.foreach { doc =>
      val termCounts = doc._2
      val (idsglobal: List[Int], cts: Array[Double]) = termCounts match {
        case v: DenseVector => ((0 until v.size).toList, v.values)
        case v: SparseVector => (v.indices.toList, v.values)
        case v => throw new IllegalArgumentException("Online LDA does not support vector type "
          + v.getClass)
      }
      val ids = idsglobal.map(globle2local) // local ids, have order

      // Initialize the variational distribution q(theta|gamma) for the mini-batch
      var gammad = new Gamma(gammaShape, 1.0 / gammaShape).samplesVector(k).t // 1 * K
    var Elogthetad = digamma(gammad) - digamma(sum(gammad)) // 1 * K
    var expElogthetad = exp(Elogthetad) // 1 * K
    val expElogbetad = expElogbeta(::, ids).toDenseMatrix // K * ids

      var phinorm = expElogthetad * expElogbetad + 1e-100 // 1 * ids
    var meanchange = 1D
      val ctsVector = new BDV[Double](cts).t // 1 * ids

      // Iterate between gamma and phi until convergence
      while (meanchange > 1e-3) {
        val lastgamma = gammad
        //        1*K                  1 * ids               ids * k
        gammad = (expElogthetad :* ((ctsVector / phinorm) * expElogbetad.t)) + alpha
        Elogthetad = digamma(gammad) - digamma(sum(gammad))
        expElogthetad = exp(Elogthetad)
        phinorm = expElogthetad * expElogbetad + 1e-100
        meanchange = sum(abs(gammad - lastgamma)) / k
      }

      gammaList += Tuple2(doc._1, gammad.t)  // K, doc topic
    val m1 = expElogthetad.t
      val m2 = (ctsVector / phinorm).t.toDenseVector
      for(i <- 0 until ids.size) {
        stat(::, ids(i)) :+=  m1 * m2(i)
      }
    }
    end = System.currentTimeMillis()
    println(s"[model train train , partition $index time ${end - start} ms]")
    start = end

    // compute new lambda and delta lambda
    val partitionResult = stat :* expElogbeta
    val lambdap = partitionResult * (corpusSize.toDouble / partitionSize.toDouble) + eta
    lambdap :-= lambda // delta lambda
    val weight = math.pow(getTau0 + iteration, -getKappa)
    lambdap :*= weight
    // push delta lambda
    for(i <- 0 until vocabs.size().toInt){
      wt(local2globle(i)) = lambdap(::, i).toArray.map(_.toFloat)
    }
    wtm.push(wt, session)
    session.progress(partitionSize)
    end = System.currentTimeMillis()
    println(s"[model train push , partition $index time ${end - start} ms]")
    start = end

    session.disconnect()

    gammaList.toIterator //gamma, doc-topic counts
  }


  override def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
    new OnlineLDAModel(Matrices.fromBreeze(lambda).transpose, this.alpha, this.gammaShape)
  }

  /**
    * Update lambda based on the batch submitted. batchSize can be different for each iteration.
    */
  private[topicModeling] def update(stat: BDM[Double], iter: Int, batchSize: Int): Unit = {
    // weight of the mini-batch.
    val weight = math.pow(getTau0 + iter, -getKappa)

    // Update lambda based on documents.
    lambda = lambda * (1 - weight) +
      (stat * (corpusSize.toDouble / batchSize.toDouble) + eta) * weight
  }

  /**
    * Get a random matrix to initialize lambda
    */
  private def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

  def topicPerplexity(m : Model, monitorPath : String): Double = {
    var topicScore = 0D
    var start = System.currentTimeMillis()
    var end = start
    // get word topic from ps in this partition
    val vocabs = new KeyList()
    for (id <- 0 until vocabSize) {
      vocabs.addKey(id)
    }
    val session = new Session(m, monitorPath, 0)
    val wtm = m.getMatrix("word-topics").asInstanceOf[FloatMatrixWithIntKey]
    val wt: mutable.HashMap[Int, Array[Float]] = wtm.fetch(vocabs, session)
    end = System.currentTimeMillis()
    println(s"[model train fetch , partition 0 time ${end - start} ms]")
    start = end

    for (i <- 0 until vocabSize) {
      val topic = wt.get(i) match {
        case Some(topicArray) => topicArray
        case None => throw new IllegalArgumentException("Convert ps word topic to matrix ")
      }
      lambda(::, i) := BDV(topic.map(_.toDouble))  // fill local lambda with ps parameter
    }

    val Elogbeta = OnlineLDAOptimizer.dirichletExpectation(lambda)

    // E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
    val sumEta = eta * vocabSize
    topicScore += sum((eta - lambda) :* Elogbeta)
    topicScore += sum(lgamma(lambda) - lgamma(eta))
    topicScore += sum(lgamma(sumEta) - lgamma(sum(lambda(::, breeze.linalg.*))))

    session.disconnect()

    topicScore
  }

  def docPerplexity(m : Model, monitorPath : String, docs: RDD[(Long, Vector)], gammaArray: RDD[(Long, BDV[Double])]): Double = {
    val docScore = docs.join(gammaArray).mapPartitionsWithIndex(docPerplexity(m, monitorPath)).sum()
    docScore
  }

  def docPerplexity(m : Model, monitorPath: String)(index: Int, it: Iterator[(Long, (Vector, BDV[Double]))])
  : Iterator[Double] = {
    val docgammas = it.toArray
    val vocabset = new mutable.HashSet[Int]
    docgammas.foreach { docgamma =>
      val termCounts = docgamma._2._1
      val (ids: List[Int], _) = termCounts match {
        case v: DenseVector => ((0 until v.size).toList, v.values)
        case v: SparseVector => (v.indices.toList, v.values)
        case v => throw new IllegalArgumentException("Online LDA does not support vector type "
          + v.getClass)
      }
      vocabset ++= ids
    }
    val vocabs = new KeyList()
    for (id <- vocabset.toArray.sorted) {
      vocabs.addKey(id)
    }

    val session = new Session(m, monitorPath, index)
    val wtm = m.getMatrix("word-topics").asInstanceOf[FloatMatrixWithIntKey]
    val wt: mutable.HashMap[Int, Array[Float]] = wtm.fetch(vocabs, session)

    val alphaVector = Vectors.dense(Array.fill(k)(alpha))
    val brzAlpha = alphaVector.toBreeze.toDenseVector
    var docScore = 0.0D
    docgammas.foreach { docgamma =>
      val (termCounts: Vector, gammad: BDV[Double]) = docgamma._2

      val Elogthetad: BDV[Double] = digamma(gammad) - digamma(sum(gammad))
      // E[log p(doc | theta, beta)]
      termCounts.foreachActive { case (idx, count) =>
        val topic = wt.get(idx) match {
          case Some(topicArray) => BDV(topicArray.map(_.toDouble))
          case None => throw new IllegalArgumentException("Convert ps word topic to matrix ")
        }
        val x = Elogthetad + topic
        val a = max(x)
        docScore += count * (a + log(sum(exp(x :- a))))
      }
      // E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
      docScore += sum((brzAlpha - gammad) :* Elogthetad)
      docScore += sum(lgamma(gammad) - lgamma(brzAlpha))
      docScore += lgamma(sum(brzAlpha)) - lgamma(sum(gammad))
    }
    session.disconnect()

    Iterator(docScore)
  }
}

