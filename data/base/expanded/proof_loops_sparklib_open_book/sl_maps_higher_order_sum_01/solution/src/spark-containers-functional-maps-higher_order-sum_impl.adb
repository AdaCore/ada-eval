--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Maps.Higher_Order)
function Sum_Impl
  (M     : Map;
   Value :
     not null access function
       (K : Key_Type; E : Element_Type) return Big_Integer)
   return Big_Integer
is
begin
   return Res : Big_Integer := 0 do
      for Submap in Iterate (M) loop
         pragma
           Loop_Invariant
             (Static => Sum_Rec (M, Value) = Res + Sum_Rec (Submap, Value));
         declare
            K : Key_Type renames Choose (Submap);
         begin
            Res := Res + Value (K, Get (Submap, K));
         end;
      end loop;
   end return;
end Sum_Impl;
