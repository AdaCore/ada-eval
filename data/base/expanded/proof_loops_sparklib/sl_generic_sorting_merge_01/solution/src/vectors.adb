--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package body Vectors with SPARK_Mode is

   procedure Insert_Space
     (Elements : in out Elements_Array;
      Last     : in out Extended_Index;
      Before   : Extended_Index;
      Count    : Count_Type := 1)
   with
     Import,
     Global     => null,
     Exit_Cases =>
       (not Valid_Index_For_Insertion (Last, Before)
        or Exceeds_Last_Count (To_Array_Index (Last), Count)               =>
          (Exception_Raised => Constraint_Error),
        Valid_Index_For_Insertion (Last, Before)
        and not Exceeds_Last_Count (To_Array_Index (Last), Count)
        and Exceeds_Capacity (To_Array_Index (Last), Elements'Last, Count) =>
          (Exception_Raised => Capacity_Error),
        others                                                             =>
          Normal_Return),
     Pre        =>
       (Static =>
          Elements'First = 1
          and then Last <= No_Index + Count_Type'Pos (Last_Count)
          and then To_Array_Index (Last) <= Elements'Last
          and then
            (for all I in 1 .. To_Array_Index (Last) =>
               Elements (I).V'Initialized)),
     Post       =>
       (Static =>
          To_Array_Index (Last) = To_Array_Index (Last'Old) + Count
          and then To_Array_Index (Last) <= Elements'Last
          and then
            (for all I in 1 .. To_Array_Index (Last'Old) =>
               Elements (I)'Initialized)
          and then
            (if Count /= 0
             then
               (for all I in 1 .. To_Array_Index (Last) =>
                  (if I < To_Array_Index (Before)
                     or else I - To_Array_Index (Before) >= Count
                   then Elements (I)'Initialized))));
   --  Insert_Space takes the backing array and last index separately, rather
   --  than a Vector, on purpose. It slides the existing elements up to open a
   --  hole of Count cells at Before and reports the grown last index in the
   --  in-out Last parameter, but it does NOT touch a Vector: a Ghost_Predicate
   --  is checked at every boundary a Vector crosses (every parameter pass and
   --  return, and after a call updates one of its components), and the hole it
   --  opens transiently breaks that predicate.
   --
   --  The caller must therefore pass a LOCAL last index (not Container.Last
   --  directly) and keep Container.Last at its old value across the call and
   --  the hole-fill: with the old last, Container's predicate is consistent
   --  with the slid array (the post keeps the old used prefix initialized), so
   --  the checks after the call and the fill hold. The caller assigns the
   --  reported new last to Container.Last only after filling the hole. The
   --  post leaves exactly the hole
   --  (To_Array_Index (Before) .. To_Array_Index (Before) + Count - 1)
   --  possibly uninitialized.

   -----------
   -- Merge --
   -----------

   procedure Merge (Target : in out Vector; Source : in out Vector)
   is separate;

end Vectors;
