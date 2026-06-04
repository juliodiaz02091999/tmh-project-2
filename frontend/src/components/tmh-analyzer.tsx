"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { Loader2, Upload, Camera } from "lucide-react"
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
  const cameraRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [showCamera, setShowCamera] = useState(false)

  useEffect(() => {
    if (showCamera && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }, [showCamera])

  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      })
      streamRef.current = stream
      setShowCamera(true)
    } catch {
      cameraRef.current?.click()
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current) return
    const canvas = document.createElement("canvas")
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) {
        handleFile(new File([blob], "foto.jpg", { type: "image/jpeg" }))
        closeCamera()
      }
    }, "image/jpeg", 0.9)
  }

  const closeCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setShowCamera(false)
  }

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
    <>
    <div className="max-w-5xl mx-auto px-4 sm:px-6 space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">TMH Analyzer</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Medición automatizada de la altura del menisco lagrimal
        </p>
      </div>

      {/* Upload */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-6 sm:p-10 text-center cursor-pointer transition-colors ${
              dragOver
                ? "border-primary bg-muted/40"
                : "hover:border-primary/50 hover:bg-muted/20"
            }`}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-48 mx-auto rounded" />
            ) : (
              <div className="space-y-2 text-muted-foreground">
                <Upload className="mx-auto h-8 w-8 sm:h-10 sm:w-10 opacity-40" />
                <p className="text-sm sm:text-base">
                  <span className="hidden sm:inline">Arrastra tu imagen aquí o </span>
                  Toca para seleccionar
                </p>
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
            <input
              ref={cameraRef}
              type="file"
              className="hidden"
              accept="image/*"
              capture="environment"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>

          <Button
            variant="outline"
            className="w-full"
            onClick={openCamera}
          >
            <Camera className="mr-2 h-4 w-4" />
            Tomar foto
          </Button>

          {file && (
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:justify-between">
              <span className="text-sm text-muted-foreground truncate min-w-0">
                {file.name}
              </span>
              <Button onClick={handleAnalyze} disabled={loading} className="w-full sm:w-auto">
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loading ? "Procesando..." : "Analizar imagen"}
              </Button>
            </div>
          )}

          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Metrics grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Card>
              <CardHeader className="pb-1 pt-3 px-3 sm:px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  TMH
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 sm:px-4 pb-3">
                <p className="text-2xl sm:text-3xl font-bold">{result.tmh_mm.toFixed(3)}</p>
                <p className="text-xs text-muted-foreground">mm</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1 pt-3 px-3 sm:px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  TMH (px)
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 sm:px-4 pb-3">
                <p className="text-2xl sm:text-3xl font-bold">{result.tmh_px.toFixed(1)}</p>
                <p className="text-xs text-muted-foreground">píxeles</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1 pt-3 px-3 sm:px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Calidad
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 sm:px-4 pb-3 flex items-center">
                <Badge variant={QUALITY_VARIANT[result.quality] ?? "outline"}>
                  {result.quality}
                </Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1 pt-3 px-3 sm:px-4">
                <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Iris Ø
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 sm:px-4 pb-3">
                <p className="text-2xl sm:text-3xl font-bold">
                  {result.iris_diameter_px.toFixed(0)}
                </p>
                <p className="text-xs text-muted-foreground">px</p>
              </CardContent>
            </Card>
          </div>

          {/* Detection badges */}
          <div className="flex gap-2 flex-wrap">
            <Badge variant={result.pupil_reflection_found ? "default" : "secondary"}>
              Reflejo pupilar {result.pupil_reflection_found ? "✓" : "✗"}
            </Badge>
            <Badge variant={result.reflection_detected ? "default" : "secondary"}>
              Reflejo menisco {result.reflection_detected ? "✓" : "✗"}
            </Badge>
          </div>

          {/* Images tabs — scrollable on mobile */}
          <Tabs defaultValue="result">
            <div className="overflow-x-auto pb-1">
              <TabsList className="w-max min-w-full">
                {Object.keys(result.images).map((key) => (
                  <TabsTrigger key={key} value={key} className="text-xs sm:text-sm">
                    {IMAGE_LABELS[key] ?? key}
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>
            {Object.entries(result.images).map(([key, src]) => (
              <TabsContent key={key} value={key}>
                <Card>
                  <CardContent className="pt-3 px-3 sm:px-6">
                    <img
                      src={src}
                      alt={IMAGE_LABELS[key] ?? key}
                      className="w-full rounded object-contain max-h-[70vh]"
                    />
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      )}
    </div>

    {/* Camera modal */}
    {showCamera && (
      <div className="fixed inset-0 z-50 bg-black flex flex-col">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="flex-1 w-full object-cover"
        />
        <div className="p-6 flex gap-4 justify-center bg-black">
          <Button variant="outline" onClick={closeCamera} className="flex-1 max-w-xs">
            Cancelar
          </Button>
          <Button onClick={capturePhoto} className="flex-1 max-w-xs">
            <Camera className="mr-2 h-4 w-4" />
            Capturar
          </Button>
        </div>
      </div>
    )}
    </>
  )
}
