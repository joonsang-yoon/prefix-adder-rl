package PrefixAdderLib

import chisel3._
import chisel3.util._

import scala.collection.mutable

class PrefixPair extends Bundle {
  val g = Bool()
  val p = Bool()
}

object PrefixPair {
  def combine(low: PrefixPair, high: PrefixPair): PrefixPair = {
    val out = Wire(new PrefixPair)
    out.g := high.g || (high.p && low.g)
    out.p := high.p && low.p
    out
  }
}

class PrefixAdderCore(val width: Int, val topology: DependentTopology) extends Module {
  require(width >= 1, s"width must be >= 1, got ${width}")
  require(
    topology.width == width,
    s"topology width ${topology.width} does not match hardware width ${width}"
  )

  val io = IO(new Bundle {
    val a = Input(UInt(width.W))
    val b = Input(UInt(width.W))
    val cin = Input(Bool())
    val sum = Output(UInt(width.W))
    val cout = Output(Bool())
  })

  val leafPairs = Wire(Vec(width, new PrefixPair))
  for (idx <- 0 until width) {
    leafPairs(idx).g := io.a(idx) && io.b(idx)
    leafPairs(idx).p := io.a(idx) ^ io.b(idx)
  }

  val memo = mutable.LinkedHashMap.empty[String, PrefixPair]
  var nextInternalId = 0

  def emit(tree: PrefixTree): PrefixPair = tree match {
    case Leaf(index) => leafPairs(index)
    case node @ Node(left, right) =>
      memo.getOrElseUpdate(
        node.signature, {
          val out = PrefixPair.combine(emit(left), emit(right))
          out.suggestName(s"prefix_node_${nextInternalId}")
          nextInternalId += 1
          out
        }
      )
  }

  val outputs = Wire(Vec(width, new PrefixPair))
  topology.outputs.zipWithIndex.foreach { case (tree, idx) =>
    outputs(idx) := emit(tree)
  }

  val carries = Wire(Vec(width + 1, Bool()))
  carries(0) := io.cin
  for (idx <- 0 until width) {
    carries(idx + 1) := outputs(idx).g || (outputs(idx).p && io.cin)
  }

  io.sum := Cat((width - 1 to 0 by -1).map(idx => leafPairs(idx).p ^ carries(idx)))
  io.cout := carries(width)
}
