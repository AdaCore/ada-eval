separate (Test)
procedure Test_Clear is
   X : List (100);
begin
   for I in 1 .. 100 loop
      Append (X, I);
      pragma Loop_Invariant (Length (X) = Count_Type (I));
   end loop;
   Assert (Length (X) = 100, "Clear does not lose any possible list cell");
end Test_Clear;
