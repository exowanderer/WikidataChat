import type { Source } from './source.js'

export interface ResponseObject {
  answer: string
  sources?: Source[]
}
