--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Maps.Higher_Order)
function Transform_Element
  (M              : Map;
   Transform_Item :
     not null access function (E : Element_Type) return Element_Type)
   return Map is
begin
   return Res : Map do
      for Submap in Iterate (M) loop

         declare
            K : Key_Type renames Choose (Submap);
         begin
            Res := Add (Res, K, Transform_Item (Get (Submap, K)));
         end;
      end loop;
   end return;
end Transform_Element;
