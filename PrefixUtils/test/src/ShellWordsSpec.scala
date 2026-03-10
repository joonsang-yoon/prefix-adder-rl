package PrefixUtils.test

import PrefixUtils._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ShellWordsSpec extends AnyFlatSpec with Matchers {
  "ShellWords.split" should "split plain whitespace-delimited commands" in {
    ShellWords.split("python -m librelane --version") shouldBe Vector(
      "python",
      "-m",
      "librelane",
      "--version"
    )
  }

  it should "preserve quoted segments and escaped whitespace" in {
    ShellWords.split("python -m \"librelane runner\" --tag sky130\\ hd") shouldBe Vector(
      "python",
      "-m",
      "librelane runner",
      "--tag",
      "sky130 hd"
    )
  }

  it should "keep empty quoted arguments" in {
    ShellWords.split("cmd '' \"\"") shouldBe Vector("cmd", "", "")
  }

  it should "reject unterminated quoted strings" in {
    an[IllegalArgumentException] should be thrownBy ShellWords.split("'unterminated")
  }
}
