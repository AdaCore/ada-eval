separate (Test_Resize)
procedure Insert_Elements is
begin
   for I in 1 .. 1000 loop
      Insert (X, I);
      pragma Loop_Invariant (Length (X) = Count_Type (I));
      pragma Loop_Invariant (for all K of X => K in 1 .. I);
   end loop;
end Insert_Elements;
