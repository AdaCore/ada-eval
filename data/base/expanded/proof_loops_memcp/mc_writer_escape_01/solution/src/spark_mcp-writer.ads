--  Spark_Mcp.Writer: the outbound JSON path's foundation -- RFC 8259 string
--  escaping, on top of which a response is a few concatenations. Pure text,
--  with no dependency on any JSON library.

package Spark_Mcp.Writer with SPARK_Mode => On is

   Max_Expansion : constant := 6;
   --  The most characters an escaped Character can occupy: a control character
   --  with no short form becomes "\u00XX".

   function Escape (S : String) return String
   with
     Pre  => S'Length <= Natural'Last / Max_Expansion,
     Post => Escape'Result'First = 1
             and then Escape'Result'Length <= Max_Expansion * S'Length;
   --  Escape S into the content of a JSON string -- what sits between the
   --  surrounding quotes -- escaping ", \ and the control characters. Bytes
   --  >= 16#20#, UTF-8 continuation bytes included, pass through unchanged.
   --  @param S The raw text to escape.
   --  @return The escaped content, ready to sit between JSON string quotes.

end Spark_Mcp.Writer;
