--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Vectors.Higher_Order)
procedure Lemma_Count_All
  (S    : Sequence;
   Last : Extended_Index;
   Test : not null access function (E : Element_Type) return Boolean)
is

   procedure Do_Proof
   with Ghost => Static;
   --  Prove the lemma

   --------------
   -- Do_Proof --
   --------------

   procedure Do_Proof is
   begin
      for I in Index_Type'First .. Last loop
         pragma
           Loop_Invariant
             (Count_Rec (S, I, Test)
              = Big (I) - Big (Extended_Index'First));
      end loop;
   end Do_Proof;

begin
   Do_Proof;
end Lemma_Count_All;
