package body Show_Ineffective_Statements is

   procedure Swap1 (X, Y : in out T) is
      Tmp : T;
   begin
      Tmp := X; -- This statement is ineffective
      X   := Y;
      Y   := X;
   end Swap1;

   Tmp : T := 0;

   procedure Swap2 (X, Y : in out T) is
      Temp : T := X; -- This variable is unused
   begin
      X := Y;
      Y := Tmp;
   end Swap2;

end Show_Ineffective_Statements;