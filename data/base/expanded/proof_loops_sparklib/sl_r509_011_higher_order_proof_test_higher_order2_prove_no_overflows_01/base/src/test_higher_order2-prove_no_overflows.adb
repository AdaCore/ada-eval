separate (Test_Higher_Order2)
procedure Prove_No_Overflows (A : Small_Array) is
   S : Integer := 0;
begin
   for I in A'Range loop
      S := S + Value (A (I));
   end loop;
end Prove_No_Overflows;
