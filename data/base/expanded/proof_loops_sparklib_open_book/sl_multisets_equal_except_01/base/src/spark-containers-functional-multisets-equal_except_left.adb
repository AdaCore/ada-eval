--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "procedure ""Lemma_Leq_Cardinality"" is not referenced");
pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Equal_Except_Left
  (Left : Multiset; Right : Multiset; Element : Element_Type) return Boolean
is

begin
   for S in Iterate (Left.Map) loop
      declare
         E : constant Element_Type := Choose (S);
      begin
         if Equivalent_Elements (E, Element) then
            null;
         elsif not Has_Key (Right.Map, E)
           or else Get (Right.Map, E) /= Get (Left.Map, E)
         then
            return False;
         end if;
      end;
   end loop;

   return True;
end Equal_Except_Left;
