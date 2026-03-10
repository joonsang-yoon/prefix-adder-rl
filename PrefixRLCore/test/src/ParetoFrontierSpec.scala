package PrefixRLCore.test

import PrefixRLCore._
import PrefixAdderLib.DependentTopology
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ParetoFrontierSpec extends AnyFlatSpec with Matchers {
  private def candidate(id: String, episode: Int, ppa: PpaTuple): EvaluatedTopology = EvaluatedTopology(
    id = id,
    episode = episode,
    topology = DependentTopology.ripple(4),
    ppa = ppa,
    topologyPath = s"${id}.json",
    rtlDir = s"rtl/${id}",
    metricsPath = Some(s"metrics/${id}.json"),
    duplicate = false
  )

  "ParetoFrontier" should "keep only non-dominated designs" in {
    val frontier = new ParetoFrontier

    frontier.update(candidate("a", 0, PpaTuple(10.0, 5.0, 100.0))).addedToFrontier shouldBe true
    frontier.update(candidate("b", 1, PpaTuple(11.0, 6.0, 101.0))).addedToFrontier shouldBe false
    frontier.update(candidate("c", 2, PpaTuple(9.0, 6.0, 101.0))).addedToFrontier shouldBe true

    frontier.snapshot.map(_.id).toSet shouldBe Set("a", "c")
  }

  it should "reject equivalent objective tuples as non-novel frontier points" in {
    val frontier = new ParetoFrontier
    val normalizer = new RunningObjectiveNormalizer
    val ppa = PpaTuple(10.0, 5.0, 100.0)

    frontier.update(candidate("a", 0, ppa)).addedToFrontier shouldBe true
    normalizer.observe(ppa)

    val preview = frontier.preview(ppa, normalizer)
    preview.isDominated shouldBe true
    frontier.update(candidate("b", 1, ppa)).addedToFrontier shouldBe false
  }
}
