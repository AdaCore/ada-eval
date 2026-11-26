--  Spark_Mcp.Writer body: escaping in one pass into a worst-case-sized buffer.

package body Spark_Mcp.Writer with SPARK_Mode => On is

   Hex_Digit : constant array (0 .. 15) of Character := "0123456789abcdef";
   --  Lowercase hex digits indexed by value, for the "\u00XX" form.

   ------------
   -- Escape --
   ------------

   function Escape (S : String) return String is separate;

end Spark_Mcp.Writer;
