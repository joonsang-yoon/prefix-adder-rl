package PrefixAdderLib.test

import PrefixAdderLib._
import PrefixUtils.Catalan
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DependentTopologySpec extends AnyFlatSpec with Matchers {
  "Catalan.dependentNetworkCount" should "follow the dependent-tree recurrence for small widths" in {
    Catalan.dependentNetworkCount(1) shouldBe BigInt(1)
    Catalan.dependentNetworkCount(2) shouldBe BigInt(1)
    Catalan.dependentNetworkCount(3) shouldBe BigInt(2)
    Catalan.dependentNetworkCount(4) shouldBe BigInt(8)
    Catalan.dependentNetworkCount(5) shouldBe BigInt(72)
    Catalan.dependentNetworkCount(6) shouldBe BigInt(1656)
  }

  it should "count the number of dependent extensions at each width" in {
    Catalan.extensionCount(1) shouldBe BigInt(1)
    Catalan.extensionCount(2) shouldBe BigInt(1)
    Catalan.extensionCount(3) shouldBe BigInt(2)
    Catalan.extensionCount(4) shouldBe BigInt(4)
    Catalan.extensionCount(5) shouldBe BigInt(9)
  }

  "DependentTopology" should "emit valid ripple and balanced dependent topologies" in {
    val ripple = DependentTopology.ripple(8)
    ripple.width shouldBe 8
    ripple.outputs.last.pretty should include("x7")

    val balanced = DependentTopology.balanced(8)
    balanced.width shouldBe 8
    balanced.outputs.last.high shouldBe 7
    balanced.stats.uniqueInternalNodes should be > 0
  }
}
