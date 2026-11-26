--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "procedure ""Lemma_Leq_Cardinality"" is not referenced");
pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Sum (Left : Multiset; Right : Multiset) return Multiset is
   E         : Element_Type;
   Set_Union : Multiset := Left;
   Buffer    : Multiset;

begin
   Buffer := Right;
   while not Is_Empty (Buffer) loop
      E := Choose (Buffer);

      Set_Union := Add (Set_Union, E, Get (Buffer.Map, E));
      Buffer := Remove_All (Buffer, E);

      pragma Loop_Variant (Decreases => Cardinality (Buffer));
   end loop;

   return Set_Union;
end Sum;
