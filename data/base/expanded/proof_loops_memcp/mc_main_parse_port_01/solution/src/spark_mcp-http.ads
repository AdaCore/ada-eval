--  Extracted from memcp, crates/spark_mcp/src/spark_mcp-http.ads.

package Spark_Mcp.Http
  with SPARK_Mode => On
is

   type Port_Number is range 1 .. 65_535;
   --  A valid TCP port the server may bind, excluding 0 (1 .. 65_535).

end Spark_Mcp.Http;
