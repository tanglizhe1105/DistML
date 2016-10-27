package org.apache.spark.mllib.topicModeling

import com.intel.distml.api.Model
import com.intel.distml.util.IntArrayWithIntKey
import com.intel.distml.util.scala.{DoubleArrayWithIntKey, DoubleMatrixWithIntKey}

/**
  * Created by Administrator on 2016/10/21.
  */
class OnlineHDPPSModel(
                  val V : Int = 0,
                  val T: Int = 20,
                  val alpha : Double = 0.01,
                  val beta : Double = 0.01
                ) extends Model {

  val alpha_sum = alpha * T
  val beta_sum = beta * V

  registerMatrix("lambda", new DoubleMatrixWithIntKey(V, T))  //word topic matrix
  registerMatrix("lambda_sum", new DoubleArrayWithIntKey(T))  //word topic matrix
  registerMatrix("var_phi", new DoubleArrayWithIntKey(T))  //var phi
}