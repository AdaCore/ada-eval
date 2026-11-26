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

--  Extracted from json-spark, json/src/json-parsers.ads.

with JSON.Types;

generic
   with package Types is new JSON.Types (<>);
package JSON.Parsers with SPARK_Mode => On is
   --  Parse JSON text into an owned tree of Types.JSON_Value nodes.
   --  @formal Types The instance of JSON.Types whose JSON_Value tree the
   --    parser builds

   pragma Preelaborate;
   pragma Elaborate_Body;

   use type Types.JSON_Value_Access;

end JSON.Parsers;
