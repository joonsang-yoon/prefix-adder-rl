package ExternalModule

import chisel3._

class RegisteredSumCarry(val width: Int) extends Module {
  val io = IO(new Bundle {
    val sumIn = Input(UInt(width.W))
    val coutIn = Input(Bool())
    val sumOut = Output(UInt(width.W))
    val coutOut = Output(Bool())
  })

  io.sumOut := RegNext(io.sumIn)
  io.coutOut := RegNext(io.coutIn)
}
