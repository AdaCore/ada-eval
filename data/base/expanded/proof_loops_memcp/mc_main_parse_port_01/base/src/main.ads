--  Extracted from memcp, src/main.adb.
--
--  Parse_Port turns the MEMCP_PORT environment value into the TCP port the
--  HTTP
--  transport binds: digits only, in 1 .. 65_535, else the default 8786.  It is
--  hand-rolled because Integer'Value raises Constraint_Error on junk, and a
--  composition root that dies on a stray environment variable is worse than
--  one
--  that falls back.
--
--  Nothing else of Main comes with it.  Parse_Port reads only its one
--  parameter, writes nothing, touches no state and calls nothing, so Main's
--  environment reader, the HOME-relative path defaults, the resource lifecycle
--  and the Spark_Mcp / Memcp.Resources / Memcp.Tools context are all dropped.
--  Only Parse_Port's return type is kept, as a trimmed Spark_Mcp.Http.

with Spark_Mcp.Http;

package Main
  with SPARK_Mode => On
is

   function Parse_Port (S : String) return Spark_Mcp.Http.Port_Number;
   --  S as a port number: digits only, in 1 .. 65_535, else the default
   --  8786. Hand-rolled rather than 'Value, which raises on junk.

end Main;
