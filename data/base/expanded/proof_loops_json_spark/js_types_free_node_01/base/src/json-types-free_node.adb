--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

pragma Warnings (Off, "* is an Ada 2022 unit");
pragma Warnings (On, "* is an Ada 2022 unit");
separate (JSON.Types)
procedure Free_Node (Object : in out JSON_Value_Access) is
   Node : JSON_Value_Access := Object;
   --  The tree, moved out so that Object is nulled on every path.
begin
   Object := null;

   while Node /= null loop
      pragma Loop_Variant (Decreases => Size (Node));

      Free_Node (Node.First_Child);
      Free_Text (Node.Key);
      Free_Text (Node.Str);

      declare
         Rest : constant JSON_Value_Access := Node.Next;
      begin
         Node.Next := null;
         Dealloc (Node);
         Node := Rest;
      end;
   end loop;
end Free_Node;
