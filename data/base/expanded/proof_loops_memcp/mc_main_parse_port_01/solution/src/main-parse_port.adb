separate (Main)
function Parse_Port (S : String) return Spark_Mcp.Http.Port_Number is
   Acc : Natural := 0;
begin
   if S'Length = 0 then
      return 8786;
   end if;
   for I in S'Range loop
      pragma Loop_Invariant (Acc <= 65_535);
      if S (I) not in '0' .. '9' then
         return 8786;
      end if;
      Acc := Acc * 10 + (Character'Pos (S (I)) - Character'Pos ('0'));
      if Acc > 65_535 then
         return 8786;
      end if;
   end loop;
   if Acc = 0 then
      return 8786;
   end if;
   return Spark_Mcp.Http.Port_Number (Acc);
end Parse_Port;
