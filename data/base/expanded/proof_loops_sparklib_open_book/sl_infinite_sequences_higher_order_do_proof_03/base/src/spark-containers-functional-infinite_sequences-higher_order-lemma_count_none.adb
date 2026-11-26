--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Infinite_Sequences.Higher_Order)
procedure Lemma_Count_None
  (S    : Sequence;
   Last : Big_Natural;
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
      for I in Interval'(1, Last) loop
         null;
      end loop;
   end Do_Proof;

begin
   Do_Proof;
end Lemma_Count_None;
