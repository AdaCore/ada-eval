package body Show_Ineffective_Statements is

   procedure Swap1 (X, Y : in out T) is
      Tmp : T := X;
   begin
      X   := Y;
      Y   := Tmp;
   end Swap1;

   procedure Swap2 (X, Y : in out T) is
      Tmp : T := Y;
   begin
      Y := X;
      X := Tmp;
   end Swap2;

end Show_Ineffective_Statements;