import chisel3.RawModule
import circt.stage.ChiselStage

import java.lang.reflect.{Constructor, InvocationTargetException, Method}
import scala.collection.mutable
import scala.util.control.NonFatal

object Elaborate {
  private val loweringOptions: String = Seq(
    "disallowLocalVariables",
    "disallowPackedArrays"
  ).mkString(",")

  private val firtoolOptions: Array[String] = Array(
    "-disable-all-randomization",
    "-strip-debug-info",
    s"--lowering-options=${loweringOptions}"
  )

  private val supportedArgTypes = "Int, Boolean, String"

  private final case class ModuleSpec(className: String, args: List[String])

  private sealed abstract class ElaborateException(message: String, cause: Throwable = null)
      extends RuntimeException(message, cause)

  private final case class ModuleParseException(message: String) extends ElaborateException(message)

  private final case class ModuleInstantiationException(message: String, cause: Throwable = null)
      extends ElaborateException(message, cause)

  def main(args: Array[String]): Unit = {
    if (args.isEmpty || args.contains("--help") || args.contains("-h")) {
      printWrapperUsage()
      ChiselStage.emitSystemVerilogFile(new RawModule {}, Array("--help"), firtoolOptions)
      sys.exit(0)
    }

    val moduleSpec = args.head
    val stageArgs = args.tail

    try {
      val outFile = ChiselStage.emitSystemVerilogFile(instantiate(moduleSpec), stageArgs, firtoolOptions)
      Console.out.println(s"Successfully generated SystemVerilog for module: ${moduleSpec}")
      Console.out.println(s"Wrote: ${outFile}")
    } catch {
      case e: ElaborateException =>
        Console.err.println(s"Error: ${e.getMessage}")
        sys.exit(1)
      case _: ClassNotFoundException =>
        Console.err.println(s"Error: Could not find class '${extractClassName(moduleSpec)}' on the classpath.")
        Console.err.println("Hint: Use a fully-qualified class name such as 'TopLevelModule.PrefixAdderMacro'.")
        sys.exit(1)
      case e: LinkageError =>
        val msg = Option(e.getMessage).getOrElse(e.toString)
        Console.err.println(s"Error: Linkage error while loading '${extractClassName(moduleSpec)}': ${msg}")
        Console.err.println("Hint: This usually means a dependency is missing or incompatible on the classpath.")
        e.printStackTrace()
        sys.exit(1)
      case NonFatal(e) =>
        val msg = Option(e.getMessage).getOrElse(e.toString)
        Console.err.println(s"Error elaborating module '${moduleSpec}': ${msg}")
        e.printStackTrace()
        sys.exit(1)
    }
  }

  private def printWrapperUsage(): Unit = {
    val msg =
      s"""|Elaborate - generate SystemVerilog from a Chisel module (CIRCT).
          |
          |Usage:
          |  bash ./mill <MillModule>.runMain Elaborate <ModuleClass>[(arg1, arg2, ...)] [ChiselStage options]
          |
          |Examples:
          |  bash ./mill TopLevelModule.runMain Elaborate TopLevelModule.ExamplePrefixAdder --target-dir generated/verilog/TopLevelModule/ExamplePrefixAdder
          |  bash ./mill TopLevelModule.runMain Elaborate 'TopLevelModule.PrefixAdderMacro(8, true, "example_topologies/dependent_balanced_8.json")' --target-dir generated/verilog/PrefixAdderMacro_8
          |
          |Supported argument types (constructors / companion apply):
          |  ${supportedArgTypes}
          |
          |Default firtool options applied (Yosys-friendly):
          |  ${firtoolOptions.mkString(" ")}
          |
          |ChiselStage options:
          |""".stripMargin
    Console.out.println(msg)
  }

  private def instantiate(moduleSpec: String): RawModule = {
    val spec = parseModuleSpec(moduleSpec)
    val clazz = Class.forName(spec.className)
    if (!classOf[RawModule].isAssignableFrom(clazz)) {
      throw ModuleInstantiationException(s"Class '${spec.className}' is not a chisel3.RawModule.")
    }
    instantiateWithArgs(clazz, spec)
  }

  private def instantiateWithArgs(clazz: Class[_], spec: ModuleSpec): RawModule = {
    val constructorMatches = clazz.getConstructors.toSeq
      .filter(_.getParameterCount == spec.args.length)
      .flatMap { ctor =>
        tryCoerceArguments(ctor.getParameterTypes.toList, spec.args).map { case (coerced, score) =>
          (score, () => ctor.newInstance(coerced: _*))
        }
      }
      .sortBy { case (score, _) => -score }

    constructorMatches.headOption match {
      case Some((_, invoke)) =>
        invoke().asInstanceOf[RawModule]
      case None =>
        instantiateViaCompanion(clazz, spec)
    }
  }

  private def instantiateViaCompanion(clazz: Class[_], spec: ModuleSpec): RawModule = {
    val companionName = s"${clazz.getName}$$"
    val companionClass =
      try {
        Class.forName(companionName)
      } catch {
        case _: ClassNotFoundException =>
          throw ModuleInstantiationException(
            s"No public constructor or companion apply matched ${spec.args.length} argument(s) for '${clazz.getName}'."
          )
      }

    val moduleField = companionClass.getField("MODULE$")
    val companion = moduleField.get(null)

    val matches = companionClass.getMethods.toSeq
      .filter(m => m.getName == "apply" && m.getParameterCount == spec.args.length)
      .flatMap { method =>
        tryCoerceArguments(method.getParameterTypes.toList, spec.args).map { case (coerced, score) =>
          (score, () => method.invoke(companion, coerced: _*))
        }
      }
      .sortBy { case (score, _) => -score }

    matches.headOption match {
      case Some((_, invoke)) =>
        invoke() match {
          case module: RawModule => module
          case other =>
            throw ModuleInstantiationException(
              s"Companion apply for '${clazz.getName}' returned ${other.getClass.getName}, not RawModule."
            )
        }
      case None =>
        throw ModuleInstantiationException(
          s"No public constructor or companion apply matched ${spec.args.length} argument(s) for '${clazz.getName}'."
        )
    }
  }

  private def parseModuleSpec(raw: String): ModuleSpec = {
    val text = raw.trim
    if (text.isEmpty) {
      throw ModuleParseException("module specification may not be empty")
    }

    val openIdx = text.indexOf('(')
    if (openIdx < 0) {
      ModuleSpec(text, Nil)
    } else {
      if (!text.endsWith(")")) {
        throw ModuleParseException(s"malformed module specification '${raw}': expected a trailing ')'")
      }
      val className = text.substring(0, openIdx).trim
      val argBlob = text.substring(openIdx + 1, text.length - 1)
      ModuleSpec(className, splitArguments(argBlob))
    }
  }

  private def splitArguments(raw: String): List[String] = {
    val args = mutable.ArrayBuffer.empty[String]
    val current = new StringBuilder

    var inSingleQuotes = false
    var inDoubleQuotes = false
    var escaping = false
    var tokenStarted = false

    def commit(): Unit = {
      if (tokenStarted) {
        args += current.toString().trim
        current.clear()
        tokenStarted = false
      }
    }

    raw.foreach { ch =>
      if (escaping) {
        current.append(ch)
        tokenStarted = true
        escaping = false
      } else if (inSingleQuotes) {
        if (ch == '\'') {
          inSingleQuotes = false
        }
        current.append(ch)
        tokenStarted = true
      } else if (inDoubleQuotes) {
        ch match {
          case '\\' =>
            current.append(ch)
            tokenStarted = true
            escaping = true
          case '"' =>
            inDoubleQuotes = false
            current.append(ch)
            tokenStarted = true
          case other =>
            current.append(other)
            tokenStarted = true
        }
      } else {
        ch match {
          case '\'' =>
            inSingleQuotes = true
            current.append(ch)
            tokenStarted = true
          case '"' =>
            inDoubleQuotes = true
            current.append(ch)
            tokenStarted = true
          case ',' =>
            commit()
          case other =>
            current.append(other)
            tokenStarted = true
        }
      }
    }

    if (escaping) {
      throw ModuleParseException(s"malformed module specification '${raw}': trailing escape")
    }
    if (inSingleQuotes || inDoubleQuotes) {
      throw ModuleParseException(s"malformed module specification '${raw}': unterminated quoted string")
    }

    commit()
    args.toList
  }

  private def tryCoerceArguments(
    parameterTypes: List[Class[_]],
    rawArgs:        List[String]
  ): Option[(Array[Object], Int)] = {
    val coerced = mutable.ArrayBuffer.empty[Object]
    var score = 0

    parameterTypes.zip(rawArgs).foreach { case (paramType, rawArg) =>
      val boxed = coerceArgument(paramType, rawArg) match {
        case Some((value, valueScore)) =>
          coerced += value
          score += valueScore
        case None =>
          return None
      }
    }

    Some((coerced.toArray, score))
  }

  private def coerceArgument(paramType: Class[_], rawArg: String): Option[(Object, Int)] = {
    if (paramType == classOf[Int] || paramType == java.lang.Integer.TYPE) {
      try Some((Integer.valueOf(parseIntLiteral(rawArg)), 3))
      catch { case _: NumberFormatException => None }
    } else if (paramType == classOf[Boolean] || paramType == java.lang.Boolean.TYPE) {
      try Some((java.lang.Boolean.valueOf(parseBooleanLiteral(rawArg)), 3))
      catch { case _: IllegalArgumentException => None }
    } else if (paramType == classOf[String]) {
      Some((parseStringLiteral(rawArg), 1))
    } else {
      None
    }
  }

  private def parseStringLiteral(raw: String): String = {
    val trimmed = raw.trim
    val quoted =
      (trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
        (trimmed.startsWith("'") && trimmed.endsWith("'"))

    val body =
      if (quoted) trimmed.substring(1, trimmed.length - 1)
      else trimmed

    val out = new StringBuilder
    var escaping = false

    body.foreach { ch =>
      if (escaping) {
        out.append(ch)
        escaping = false
      } else if (ch == '\\') {
        escaping = true
      } else {
        out.append(ch)
      }
    }

    if (escaping) {
      throw ModuleParseException(s"malformed string literal '${raw}': trailing escape")
    }

    out.toString()
  }

  private def parseIntLiteral(raw: String): Int = {
    val trimmed = raw.trim
    if (trimmed.startsWith("0x") || trimmed.startsWith("0X")) {
      Integer.parseUnsignedInt(trimmed.drop(2), 16)
    } else if (trimmed.startsWith("0b") || trimmed.startsWith("0B")) {
      Integer.parseUnsignedInt(trimmed.drop(2), 2)
    } else {
      trimmed.toInt
    }
  }

  private def parseBooleanLiteral(raw: String): Boolean = raw.trim.toLowerCase match {
    case "true" | "t" | "1"  => true
    case "false" | "f" | "0" => false
    case other               => throw new IllegalArgumentException(s"expected boolean literal, found '${other}'")
  }

  private def extractClassName(moduleSpec: String): String = {
    val idx = moduleSpec.indexOf('(')
    if (idx < 0) moduleSpec else moduleSpec.substring(0, idx)
  }
}
