package body Show_Ineffective_Statements is

   procedure Swap2 (X, Y : in out T) is
      Tmp : T := Y;
   begin
      Y := X;
      X := Tmp;
   end Swap2;

end Show_Ineffective_Statements;
