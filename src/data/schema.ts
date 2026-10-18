import { z } from "zod";
import { JOINTS, JOINT_NAMES, type JointName } from "./channels";

// A joint is written as a map of its fields ({ flex: 90, abd: 20 }); joints
// with a single degree of freedom also accept a bare number (knee_l: 90).
const jointSchema = (joint: JointName) => {
  const fields = JOINTS[joint];
  const obj = z.strictObject(Object.fromEntries(fields.map((f) => [f, z.number().optional()])));
  return fields.length === 1 ? z.union([z.number(), obj]) : obj;
};

export type JointValue = number | Partial<Record<string, number>>;

const jointShape = Object.fromEntries(JOINT_NAMES.map((j) => [j, jointSchema(j).optional()])) as unknown as Record<
  JointName,
  z.ZodOptional<z.ZodType<JointValue>>
>;

/** A body shape: an optional named pose to start from plus per-joint overrides. */
export const PoseBodySchema = z.strictObject({
  pose: z.string().optional(),
  ...jointShape,
});
export type PoseBody = z.infer<typeof PoseBodySchema>;

export const KeyframeSchema = z.strictObject({
  t: z.number().min(0),
  label: z.string().optional(),
  pose: z.string().optional(),
  ...jointShape,
});
export type Keyframe = z.infer<typeof KeyframeSchema>;

const floorPoint = z.tuple([z.number(), z.number()]);

export const CATEGORIES = ["fundamento", "esquiva", "movimentação", "ataque circular", "ataque direto", "ataque de proximidade", "desequilibrante"] as const;

/** What a movement file says about the movement, for people: the app's info panel shows it. */
const textShape = {
  id: z.string().regex(/^[a-z0-9-]+$/),
  name: z.string(),
  aka: z.array(z.string()).default([]),
  translation: z.string(),
  category: z.enum(CATEGORIES),
  summary: z.string(),
  steps: z.array(z.string()).min(1),
  tips: z.array(z.string()).default([]),
  sources: z.array(z.strictObject({ title: z.string(), url: z.url() })).min(1),
};
export const MovementTextSchema = z.object(textShape);
export type MovementText = z.infer<typeof MovementTextSchema>;

/** A whole movement file: the text plus the keyframes the ML references start from (scripts/sample-clips.ts). */
export const MovementSchema = z
  .strictObject({
    ...textShape,
    side: z.enum(["right", "left"]).describe("which leg/arm the authored keyframes use"),
    loop: z.boolean().default(false),
    duration: z.number().positive().optional(),
    rest: z.number().min(0).default(0.4),
    keyframes: z.array(KeyframeSchema).min(2),
    /** Floor triangle the feet step on, [x, z] in metres: front left, front right, back. */
    triangle: z.tuple([floorPoint, floorPoint, floorPoint]).optional(),
  })
  .superRefine((m, ctx) => {
    m.keyframes.forEach((k, i) => {
      if (i > 0 && k.t <= m.keyframes[i - 1].t)
        ctx.addIssue({ code: "custom", path: ["keyframes", i, "t"], message: "keyframe times must increase" });
    });
    if (m.keyframes[0].t !== 0) ctx.addIssue({ code: "custom", path: ["keyframes", 0, "t"], message: "first keyframe must be at t: 0" });
    const last = m.keyframes[m.keyframes.length - 1].t;
    if (m.loop && !(m.duration && m.duration > last))
      ctx.addIssue({ code: "custom", path: ["duration"], message: "a looping movement needs a duration past its last keyframe" });
  });
export type Movement = z.infer<typeof MovementSchema>;

/** Library entries may also start from the mirror image of another pose. */
export const PoseEntrySchema = PoseBodySchema.extend({ mirror: z.string().optional() }).refine(
  (p) => !(p.pose && p.mirror),
  "use either pose or mirror, not both",
);
export type PoseEntry = z.infer<typeof PoseEntrySchema>;

export const PoseLibrarySchema = z.strictObject({
  poses: z.record(z.string(), PoseEntrySchema),
});
export type PoseLibrary = z.infer<typeof PoseLibrarySchema>;

export const IndexSchema = z.strictObject({
  movements: z.array(z.string()).min(1),
});
