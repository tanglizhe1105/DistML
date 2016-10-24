package org.apache.spark.mllib.topicModeling

import com.intel.distml.api.Model
import com.intel.distml.util.IntMatrixWithIntKey

/**
  * Created by Administrator on 2016/10/21.
  */
class OnlineHDPPSModel(
                  val V : Int = 0,
                  val K: Int = 20,
                  val alpha : Double = 0.01,
                  val beta : Double = 0.01
                ) extends Model {

  val alpha_sum = alpha * K
  val beta_sum = beta * V

  registerMatrix("word-topics", new IntMatrixWithIntKey(V, K))  //word topic matrix
}