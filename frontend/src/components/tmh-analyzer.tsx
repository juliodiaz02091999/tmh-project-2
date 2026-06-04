"use client"

import { useState, useCallback, useRef } from "react"
import { Loader2, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

type AnalysisResult = {
  tmh_px: number
  tmh_mm: number
  quality: string
  measurement_mode: string
  pupil_reflection_found: boolean
  reflection_detected: boolean
  iris_diameter_px: number
  images: Record<string, string>
}

const IMAGE_LABELS: Record<string, string> = {
  original: "Original",
  segmentation: "Segmentación",
  roi: "ROI",
  candidates: "Candidatos",
  result: "Resultado TMH",
}

const QUALITY_VARIANT: Record<string, "default" | "secondary" | "destructive"> = {
  OK: "default",
  REVISAR: "secondary",
  NO_CONFIABLE: "destructive",
}

export function TmhAnalyzer() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((f: File) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const f = e.dataTransfer.files[0]
      if (f?.type.startsWith("image/")) handleFile(f)
    },
    [handleFile]
  )

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append("file", file)
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
      const res = await fetch(`${apiUrl}/api/analyze`, {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? "Error en el análisis")
      }
      setResult(await res.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error desconocido")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">TMH Analyzer</h1>
        <p className="text-muted-foreground mt-1">
          Medición automatizada de la altura del menisco lagrimal
        </p>
      </div>

      <Card>
        <CardContent className="pt-6 space-y-4">
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
              dragOver
                ? "border-primary bg-muted/40"
                : "hover:border-primary/50 hover:bg-muted/20"
            }`}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-52 mx-auto rounded" />
            ) : (
              <div className="space-y-2 text-muted-foreground">
                <Upload className="mx-auto h-10 w-10 opacity-40" />
                <p className="text-base">Arrastra tu imagen aquí</p>
                <p className="text-sm">o haz clic para seleccionar</p>
                <p className="text-xs opacity-60">PNG · JPG · BMP · TIFF</p>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>

          {file && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground truncate max-w-xs">
                {file.name}
              </span>
              <Button onClick={handleAnalyze} disabled={loading}>
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loading ? "Procesando..." : "Analizar imagen"}
              </Button>
            </div>
          )}

          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-1 pt-4 px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  TMH
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <p className="text-3xl font-bold">{result.tmh_mm.toFixed(3)}</p>
                <p className="text-xs text-muted-foreground">mm</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1 pt-4 px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  TMH (px)
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <p className="text-3xl font-bold">{result.tmh_px.toFixed(1)}</p>
                <p className="text-xs text-muted-foreground">píxeles</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1 pt-4 px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Calidad
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 flex items-center">
                <Badge variant={QUALITY_VARIANT[result.quality] ?? "outline"}>
                  {result.quality}
                </Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1 pt-4 px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Iris Ø
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <p className="text-3xl font-bold">
                  {result.iris_diameter_px.toFixed(0)}
                </p>
                <p className="text-xs text-muted-foreground">px</p>
              </CardContent>
            </Card>
          </div>

          <div className="flex gap-2 flex-wrap">
            <Badge variant={result.pupil_reflection_found ? "default" : "secondary"}>
              Reflejo pupilar {result.pupil_reflection_found ? "✓" : "✗"}
            </Badge>
            <Badge variant={result.reflection_detected ? "default" : "secondary"}>
              Reflejo menisco {result.reflection_detected ? "✓" : "✗"}
            </Badge>
          </div>

          <Tabs defaultValue="result">
            <TabsList>
              {Object.keys(result.images).map((key) => (
                <TabsTrigger key={key} value={key}>
                  {IMAGE_LABELS[key] ?? key}
                </TabsTrigger>
              ))}
            </TabsList>
            {Object.entries(result.images).map(([key, src]) => (
              <TabsContent key={key} value={key}>
                <Card>
                  <CardContent className="pt-4">
                    <img
                      src={src}
                      alt={IMAGE_LABELS[key] ?? key}
                      className="w-full rounded object-contain max-h-[600px]"
                    />
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      )}
    </div>
  )
}
