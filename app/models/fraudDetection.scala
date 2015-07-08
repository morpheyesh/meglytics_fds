/*
** Copyright [2015-2016] [Megam Systems]
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
** http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/
package models

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import com.typesafe.config._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD

/**
* @author morpheyesh
*
*/

object fraudDetection {

   def Detect(): String = {
     play.api.Logger.debug("Entered the Meglytics Engine")
     val numIterations = 10

     /*
      * SparkConf & SparkContext initialization.
      */
     val conf = new SparkConf().setMaster("spark://103.56.92.29:7077").setAppName("meglytics_fds")
     val sc = new SparkContext(conf)
     sc.addJar("target/scala-2.10/meglytics_fds_2.10-1.0-SNAPSHOT.jar")

   /*
    * Getting the data from FileSystem
    */
    val rawData = sc.textFile("/home/yeshwanth/fraud_data.tsv")
     val record = rawData.map(t => t.split("\t"))
     val sampleRDD = record.first()

     play.api.Logger.debug(sampleRDD(0))
     val data = record.map {r =>
      val trim = r.map(_.replaceAll("\"", ""))
      val label = r.size - 1.toInt
      val features = trim.slice(0, r.size - 1).map(d => d.toDouble)
      LabeledPoint(label, Vectors.dense(features))}

     data.cache

     play.api.Logger.debug("Training a model")

    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
    val dataPt = data.first()

    play.api.Logger.debug("Going to predict")

    val prediction = lrModel.predict(dataPt.features)

    play.api.Logger.debug("Prediction is done. Voila!")
    play.api.Logger.debug(prediction.toString)

    prediction.toString
  }


}
