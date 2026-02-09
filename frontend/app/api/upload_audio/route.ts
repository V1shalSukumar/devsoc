import { NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";

export async function POST(req: Request) {
  const formData = await req.formData();
  const file = formData.get("audio") as File | null;

  if (!file) {
    return NextResponse.json(
      { error: "No audio file uploaded" },
      { status: 400 }
    );
  }

  // Convert file to buffer
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);

  // YOUR WINDOWS FOLDER
  const uploadPath = "C:\\Users\\hrbrv\\devsoc\\data\\uploads";

  // Full path to save the file
  const filePath = path.join(uploadPath, file.name);

  // Write file
  await writeFile(filePath, buffer);

  return NextResponse.json({
    success: true,
    path: filePath,
  });
}