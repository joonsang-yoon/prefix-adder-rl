package PrefixRLCore.test

import PrefixRLCore._
import PrefixAdderLib.DependentTopology
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SyntheticEvaluatorSpec extends AnyFlatSpec with Matchers {
  "SyntheticEvaluator" should "persist deterministic surrogate metrics and placeholder artifacts" in {
    val root = os.temp.dir(prefix = "synthetic_evaluator_spec_")
    val evaluator = new SyntheticEvaluator(
      SyntheticBackendConfig(
        workRoot = root,
        registerOutputs = true,
        clockPeriod = 5.0
      )
    )
    val topology = DependentTopology.balanced(8)

    val first = evaluator.evaluate(0, topology)
    val second = evaluator.evaluate(1, topology)

    first.ppa shouldBe second.ppa
    os.exists(first.topologyPath) shouldBe true
    os.exists(first.metricsPath.get) shouldBe true
    os.exists(first.rtlDir / "README.txt") shouldBe true

    val json = ujson.read(os.read(first.metricsPath.get))
    json("backend").str shouldBe "synthetic"
    json("mode").str shouldBe "surrogate"
    json("register_outputs").bool shouldBe true
    json("metrics")("power__total").num shouldBe first.ppa.power +- 1e-9
    json("metrics")("design__instance__area").num shouldBe first.ppa.area +- 1e-9
    (json("metrics")("clock_period").num - json("metrics")("timing__setup__ws").num) shouldBe first.ppa.delay +- 1e-9
  }

  it should "reflect output-register settings in the surrogate metrics" in {
    val root = os.temp.dir(prefix = "synthetic_evaluator_register_spec_")
    val topology = DependentTopology.ripple(8)

    val registered = new SyntheticEvaluator(
      SyntheticBackendConfig(
        workRoot = root / "registered",
        registerOutputs = true,
        clockPeriod = 5.0
      )
    ).evaluate(0, topology).ppa

    val combinational = new SyntheticEvaluator(
      SyntheticBackendConfig(
        workRoot = root / "combinational",
        registerOutputs = false,
        clockPeriod = 5.0
      )
    ).evaluate(0, topology).ppa

    registered.power should be > combinational.power
    registered.delay should be < combinational.delay
    registered.area should be > combinational.area
  }
}
