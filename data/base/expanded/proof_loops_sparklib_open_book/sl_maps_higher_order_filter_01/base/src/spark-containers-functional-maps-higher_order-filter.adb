--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Maps.Higher_Order)
function Filter
  (M    : Map;
   Test :
     not null access function
       (K : Key_Type; E : Element_Type) return Boolean) return Map is
begin
   return Res : Map do
      for Submap in Iterate (M) loop

         declare
            K : Key_Type renames Choose (Submap);
         begin
            if Test (K, Get (Submap, K)) then
               Res := Add (Res, K, Get (Submap, K));
            end if;
         end;
      end loop;
   end return;
end Filter;
