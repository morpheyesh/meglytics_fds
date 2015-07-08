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

import scalaz._
import Scalaz._
import scalaz.EitherT._
import scalaz.Validation
import scalaz.NonEmptyList._
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import org.apache.hadoop.conf._
import org.apache.hadoop.fs._

object HDFSFileService {
  private val conf = new Configuration()
  private val hdfsCoreSitePath = new Path("core-site.xml")
  private val hdfsHDFSSitePath = new Path("hdfs-site.xml")

  conf.addResource(hdfsCoreSitePath)
  conf.addResource(hdfsHDFSSitePath)

  private val fileSystem = FileSystem.get(conf)

  def saveFile(filepath: String): ValidationNel[Throwable, String] = {
    (Validation.fromTryCatch[String] {
      
      val file = new File(filepath)
      val out = fileSystem.create(new Path(file.getName))
      
      
      val in = new BufferedInputStream(new FileInputStream(file))
      var b = new Array[Byte](1024)
      var numBytes = in.read(b)
      while (numBytes > 0) {
        out.write(b, 0, numBytes)
        numBytes = in.read(b)
      }
      
      in.close()
      out.close()
      "File Uploaded"
    } leftMap { t: Throwable => nels(t) })

  }
 

  def removeFile(filename: String): Boolean = {
    val path = new Path(filename)
    fileSystem.delete(path, true)
  }

  def getFile(filename: String): InputStream = {
    val path = new Path(filename)
    fileSystem.open(path)
  }

  def createFolder(folderPath: String): Unit = {
    val path = new Path(folderPath)
    if (!fileSystem.exists(path)) {
      fileSystem.mkdirs(path)
    }
  }
}
