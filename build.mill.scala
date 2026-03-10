import mill._
import mill.scalalib._
import mill.scalalib.TestModule.ScalaTest
import mill.scalalib.scalafmt.ScalafmtModule

trait ChiselModule extends ScalaModule with ScalafmtModule {
  override def scalaVersion = "2.13.16"

  override def scalacOptions = Seq(
    "-language:reflectiveCalls",
    "-deprecation",
    "-feature",
    "-unchecked",
    "-Xcheckinit"
  )

  override def mvnDeps = Seq(
    mvn"org.chipsalliance::chisel:7.5.0",
    mvn"com.lihaoyi::os-lib:0.10.2",
    mvn"com.lihaoyi::ujson:3.1.3",
    mvn"com.lihaoyi::mainargs:0.5.0"
  )

  override def scalacPluginMvnDeps = Seq(
    mvn"org.chipsalliance:::chisel-plugin:7.5.0"
  )

  object test extends ScalaTests with ScalaTest with ScalafmtModule {
    override def mvnDeps = super.mvnDeps() ++ Seq(
      mvn"org.scalatest::scalatest:3.2.19"
    )
  }
}

object TopLevelModule extends ChiselModule {
  override def moduleDeps = Seq(ExternalModule, PrefixAdderLib, PrefixUtils)
}

object ExternalModule extends ChiselModule

object PrefixUtils extends ChiselModule

object PrefixAdderLib extends ChiselModule {
  override def moduleDeps = Seq(PrefixUtils)
}

object PrefixRLCore extends ChiselModule {
  override def moduleDeps = Seq(PrefixAdderLib, PrefixUtils)
}

object PrefixTabularRL extends ChiselModule {
  override def moduleDeps = Seq(PrefixRLCore, PrefixAdderLib, PrefixUtils)
}

object PrefixDeepRL extends ChiselModule {
  override def moduleDeps = Seq(PrefixRLCore, PrefixAdderLib, PrefixUtils)
}
