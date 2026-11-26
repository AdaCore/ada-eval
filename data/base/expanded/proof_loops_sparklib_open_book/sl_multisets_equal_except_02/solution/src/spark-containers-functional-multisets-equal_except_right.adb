--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "procedure ""Lemma_Leq_Cardinality"" is not referenced");
pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Equal_Except_Right
  (Left : Multiset; Right : Multiset; Element : Element_Type) return Boolean
is

begin
   for S in Iterate (Right.Map) loop
      pragma Loop_Invariant (Static => S <= Right.Map);
      pragma
        Loop_Invariant
          (Static =>
             (for all E of Right.Map =>
                Has_Key (S, E)
                or else Equivalent_Elements (E, Element)
                or else
                  (Has_Key (Left.Map, E)
                   and then Get (Right.Map, E) = Get (Left.Map, E))));
      declare
         E : constant Element_Type := Choose (S);
      begin
         if Equivalent_Elements (E, Element) then
            null;
         elsif not Has_Key (Left.Map, E) then
            pragma
              Assert
                (Static =>
                   Nb_Occurence (Left, E) /= Nb_Occurence (Right, E));
            return False;
         else
            pragma
              Assert (Static => Get (Right.Map, E) = Get (Left.Map, E));
         end if;
      end;
   end loop;

   return True;
end Equal_Except_Right;
