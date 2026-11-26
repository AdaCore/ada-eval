separate (Test_Higher_Order3)
procedure Prove_No_Overflows (A : Small_Matrix) is
   S : Integer := 0;
begin
   if A'Length (2) = 0 then
      return;
   end if;
   for I in A'Range (1) loop
      for J in A'Range (2) loop
         S := S + Value (A (I, J));
      end loop;
   end loop;
end Prove_No_Overflows;
