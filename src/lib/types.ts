import type { LngLatLike, LngLatBoundsLike } from 'maplibre-gl';

export type PPMGeoJSONFeatureCollection = {
  type: 'FeatureCollection';
  features: PPMGeoJSONFeature[];
};

export type PPMGeoJSONFeature = {
  type: 'Feature';
  properties: {
    title: string;
    color: 'blue' | 'green' | 'yellow' | 'red';
    centroid: LngLatLike;
    bbox: LngLatBoundsLike;
    ppmItemIDs: string[];
  };
  geometry: {
    type: 'Polygon';
    coordinates: number[][][];
  };
  id: string;
};

// Search results that share a location, shown as one marker on the map
export type ResultMarker = {
  name: string;
  lngLat: [number, number];
  items: { id: string; label: string }[];
};

export type PPMItem = {
  id: string;
  imageURL: string;
  caption: string;
  caption_en: string;
  volume: number;
  page: number;
  location: {
    regio: number;
    insula: number;
    property: string;
    room: string | null;
    geojson: PPMGeoJSONFeatureCollection;
  };
  imageVector?: number[];
  captionVector?: number[];
};
