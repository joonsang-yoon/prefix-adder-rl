package PrefixAdderLib.test

import PrefixAdderLib._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DependentTopologyJsonSpec extends AnyFlatSpec with Matchers {
  "DependentTopology" should "round-trip through JSON while preserving statistics" in {
    val topology = DependentTopology.balanced(8)
    val roundTripped = DependentTopology.fromJsonString(topology.toPrettyJson)

    roundTripped shouldBe topology
    roundTripped.stats.fingerprint shouldBe topology.stats.fingerprint
    roundTripped.stats.reuseRatio shouldBe topology.stats.reuseRatio +- 1e-12
  }

  it should "emit Graphviz DOT containing all output roots" in {
    val topology = DependentTopology.ripple(4)
    val dot = topology.toDot

    dot should include("digraph DependentPrefixTopology")
    dot should include("out0")
    dot should include("out3")
  }

  it should "reject malformed dependent-tree JSON that skips the dependent root reuse rule" in {
    val malformed =
      """{
        |  "model": "dependent-tree",
        |  "width": 4,
        |  "outputs": [
        |    {"leaf": 0},
        |    {"node": [{"leaf": 0}, {"leaf": 1}]},
        |    {"node": [
        |      {"node": [{"leaf": 0}, {"leaf": 1}]},
        |      {"leaf": 2}
        |    ]},
        |    {"node": [
        |      {"node": [
        |        {"leaf": 0},
        |        {"node": [{"leaf": 1}, {"leaf": 2}]}
        |      ]},
        |      {"leaf": 3}
        |    ]}
        |  ]
        |}""".stripMargin

    an[IllegalArgumentException] should be thrownBy DependentTopology.fromJsonString(malformed)
  }
}
