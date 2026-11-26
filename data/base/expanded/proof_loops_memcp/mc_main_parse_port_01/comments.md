Loop 1 of `Main.Parse_Port` comes from `src/main.adb`, lines 66-75.
The source revision is `8701f4902513b6e2d350c179ba24a883993323c7` in https://github.com/manthonyaiello/memcp. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses.

The reference solution proves. In `base`, removing 1 loop invariant leaves 2 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/main.ads`

Upstream Parse_Port is a function nested in the *procedure* Main -- memcp's composition root -- declared with its comment in Main's declarative part and defined further down.  A subprogram nested in a procedure body cannot be a compilation unit, so here Main is a *package* of the same name, with Parse_Port promoted to its spec so that the example has a visible entry point.  The name and the declaration travel unchanged.

### `src/spark_mcp-http.ads`

Only Port_Number, which is Parse_Port's return type.  Upstream this package also declares the transport's message-pointer type, its deallocation instance, a size constant, an exception, and an external Abstract_State modelling the socket world; none of that is reachable from the subject loop. The abstract state in particular is dropped rather than modelled: the loop neither reads nor writes it, so there is no framing obligation to lose.

### `src/spark_mcp.ads`

spark_mcp: a reusable, transport-agnostic MCP server core. Only the root of the hierarchy is needed here, so that the child unit name Spark_Mcp.Http -- and with it the spelling of Parse_Port's return type -- is the upstream one. Its own constants, error codes and response-pointer type are not reachable from the subject loop and are not copied.
