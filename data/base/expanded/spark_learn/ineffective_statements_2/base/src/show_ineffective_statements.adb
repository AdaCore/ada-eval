package body Show_Ineffective_Statements is

   Tmp : T := 0;

   procedure Swap2 (X, Y : in out T) is
      Temp : T := X;
   begin
      X := Y;
      Y := Tmp;
   end Swap2;

end Show_Ineffective_Statements;
