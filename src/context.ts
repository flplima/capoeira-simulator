import { createActorContext } from "@xstate/react";
import { capoeiraMachine } from "./machine";

export const CapoeiraContext = createActorContext(capoeiraMachine);
