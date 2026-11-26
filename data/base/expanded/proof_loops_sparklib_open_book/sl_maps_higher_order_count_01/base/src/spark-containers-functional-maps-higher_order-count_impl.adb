--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Maps.Higher_Order)
function Count_Impl
  (M    : Map;
   Test :
     not null access function
       (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
is
begin
   return Res : Big_Natural := 0 do
      for Submap in Iterate (M) loop
         declare
            K : Key_Type renames Choose (Submap);
         begin
            if Test (K, Get (Submap, K)) then
               Res := Res + 1;
            end if;
         end;
      end loop;
   end return;
end Count_Impl;
