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

package body JSON.Types
  with SPARK_Mode => On
is

   function To_Text (Value : String) return not null Streams.Text_Access;

   function To_Text (Value : String) return not null Streams.Text_Access is
      subtype Constrained_String is String (1 .. Value'Length);
   begin
      --  The conversion slides the bounds of Value to 1 .. Value'Length
      return new Constrained_String'(Constrained_String (Value));
   end To_Text;

   function Kind
     (Object : not null access constant JSON_Value) return Value_Kind
   is (Object.Kind);

   function Length
     (Object : not null access constant JSON_Value) return Natural
   is (Object.Length);

   function Is_Standalone
     (Object : not null access constant JSON_Value) return Boolean
   is (Object.Next = null and Object.Key = null);

   procedure Insert
     (Object : not null access JSON_Value;
      Key    : String;
      Value  : in out JSON_Value_Access) is separate;

end JSON.Types;
