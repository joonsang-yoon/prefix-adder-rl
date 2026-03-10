package TopLevelModule

import ExternalModule.RegisteredSumCarry
import PrefixAdderLib.{DependentTopology, PrefixAdderCore}
import chisel3._

class PrefixAdderMacro(
  width:           Int,
  registerOutputs: Boolean,
  topologyPath:    String
) extends Module {
  private val resolvedTopologyPath = os.Path(topologyPath, os.pwd)
  private val topology = DependentTopology.fromFile(resolvedTopologyPath)

  require(width >= 1, s"width must be >= 1, got ${width}")
  require(
    topology.width == width,
    s"topology width ${topology.width} does not match PrefixAdderMacro width ${width}"
  )

  val io = IO(new Bundle {
    val a = Input(UInt(width.W))
    val b = Input(UInt(width.W))
    val cin = Input(Bool())
    val sum = Output(UInt(width.W))
    val cout = Output(Bool())
  })

  val core = Module(new PrefixAdderCore(width, topology))
  core.io.a := io.a
  core.io.b := io.b
  core.io.cin := io.cin

  if (registerOutputs) {
    val outRegs = Module(new RegisteredSumCarry(width))
    outRegs.io.sumIn := core.io.sum
    outRegs.io.coutIn := core.io.cout
    io.sum := outRegs.io.sumOut
    io.cout := outRegs.io.coutOut
  } else {
    io.sum := core.io.sum
    io.cout := core.io.cout
  }
}

object PrefixAdderMacro {
  def apply(width: Int, topologyPath: String): PrefixAdderMacro =
    new PrefixAdderMacro(width, registerOutputs = true, topologyPath = topologyPath)
}

class ExamplePrefixAdder
    extends PrefixAdderMacro(
      width = 8,
      registerOutputs = true,
      topologyPath = "example_topologies/dependent_balanced_8.json"
    )
