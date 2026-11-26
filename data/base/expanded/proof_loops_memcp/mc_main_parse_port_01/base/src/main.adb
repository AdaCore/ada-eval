package body Main
  with SPARK_Mode => On
is

   ----------------
   -- Parse_Port --
   ----------------

   function Parse_Port (S : String) return Spark_Mcp.Http.Port_Number
   is separate;

end Main;
