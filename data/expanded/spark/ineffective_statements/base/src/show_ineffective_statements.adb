package body Show_Ineffective_Statements is

   procedure Swap1 (X, Y : in out T) is
      Tmp : T;
   begin
      Tmp := X;
      X   := Y;
      Y   := X;
   end Swap1;

end Show_Ineffective_Statements;