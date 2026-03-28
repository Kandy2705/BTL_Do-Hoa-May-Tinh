"""
Lighting management utilities for Phong shading.

Provides Light, Material, and LightingManager classes to simplify
lighting setup and reduce code duplication.
"""
import numpy as np
from typing import Optional, Tuple, Union


class Light:
    """Represents a light source with diffuse, specular, and ambient components."""
    
    def __init__(self, 
                 diffuse: Union[Tuple[float, float, float], np.ndarray] = (0.9, 0.4, 0.6),
                 specular: Union[Tuple[float, float, float], np.ndarray] = (0.9, 0.4, 0.6),
                 ambient: Union[Tuple[float, float, float], np.ndarray] = (0.9, 0.4, 0.6),
                 position: Union[Tuple[float, float, float], np.ndarray] = (0, 0.5, 0.9)):
        """
        Initialize a light source.
        
        Args:
            diffuse: RGB color for diffuse component (default: pinkish)
            specular: RGB color for specular component (default: pinkish)
            ambient: RGB color for ambient component (default: pinkish)
            position: 3D position of the light (default: above and in front)
        """
        self.diffuse = np.array(diffuse, dtype=np.float32)
        self.specular = np.array(specular, dtype=np.float32)
        self.ambient = np.array(ambient, dtype=np.float32)
        self.position = np.array(position, dtype=np.float32)
    
    @classmethod
    def default(cls):
        """Create a default light source."""
        return cls()


class Material:
    """Represents material properties for Phong shading."""
    
    def __init__(self,
                 diffuse: Union[Tuple[float, float, float], np.ndarray] = (0.6, 0.4, 0.7),
                 specular: Union[Tuple[float, float, float], np.ndarray] = (0.6, 0.4, 0.7),
                 ambient: Union[Tuple[float, float, float], np.ndarray] = (0.6, 0.4, 0.7),
                 shininess: float = 100.0):
        """
        Initialize material properties.
        
        Args:
            diffuse: RGB color for diffuse reflection (default: purple)
            specular: RGB color for specular reflection (default: purple)
            ambient: RGB color for ambient reflection (default: purple)
            shininess: Shininess exponent for specular highlights (default: 100.0)
        """
        self.diffuse = np.array(diffuse, dtype=np.float32)
        self.specular = np.array(specular, dtype=np.float32)
        self.ambient = np.array(ambient, dtype=np.float32)
        self.shininess = float(shininess)
    
    @classmethod
    def default(cls):
        """Create a default material."""
        return cls()


class LightingManager:
    """
    Manages lighting setup for shader programs.
    
    Reduces code duplication by centralizing lighting uniform uploads.
    """
    
    # Default values matching the original codebase
    DEFAULT_LIGHT = Light(
        diffuse=(0.9, 0.4, 0.6),
        specular=(0.9, 0.4, 0.6),
        ambient=(0.9, 0.4, 0.6),
        position=(0, 0.5, 0.9)
    )
    
    DEFAULT_MATERIAL = Material(
        diffuse=(0.6, 0.4, 0.7),
        specular=(0.6, 0.4, 0.7),
        ambient=(0.6, 0.4, 0.7),
        shininess=100.0
    )
    
    def __init__(self, uma):
        """
        Initialize lighting manager.
        
        Args:
            uma: UManager instance for uploading uniforms
        """
        self.uma = uma
    
    def setup_phong(self, 
                    light: Optional[Light] = None,
                    material: Optional[Material] = None,
                    mode: int = 1,
                    view_matrix: Optional[np.ndarray] = None,
                    display_mode: int = 0,
                    cam_far: float = 100.0):
        """
        Setup Phong lighting uniforms.
        
        Args:
            light: Light source (uses default if None)
            material: Material properties (uses default if None)
            mode: Rendering mode (default: 1)
            view_matrix: View matrix for transforming light to view space
            display_mode: 0=RGB, 1=Depth Map
            cam_far: Camera far plane for depth normalization
        """
        light = light or self.DEFAULT_LIGHT
        material = material or self.DEFAULT_MATERIAL
        
        # Create I_light matrix: [diffuse, specular, ambient]
        I_light = np.array([
            light.diffuse,
            light.specular,
            light.ambient
        ], dtype=np.float32)
        
        # Create K_materials matrix: [diffuse, specular, ambient]
        K_materials = np.array([
            material.diffuse,
            material.specular,
            material.ambient
        ], dtype=np.float32)
        
        # Upload uniforms
        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light.position, 'light_pos_world')
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(material.shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        
        # Upload view matrix for light transform
        if view_matrix is not None:
            self.uma.upload_uniform_matrix4fv(view_matrix, 'view', True)
        
        # Upload display mode and camera far plane
        self.uma.upload_uniform_scalar1i(display_mode, 'u_display_mode')
        self.uma.upload_uniform_scalar1f(cam_far, 'u_cam_far')
    
    def setup_gouraud(self,
                     light: Optional[Light] = None,
                     material: Optional[Material] = None,
                     shininess: float = 100.0,
                     view_matrix: Optional[np.ndarray] = None):
        """
        Setup Gouraud lighting uniforms (for vertex shader).
        
        Gouraud shading computes lighting per-vertex in the vertex shader,
        then interpolates the result in the fragment shader.
        
        Args:
            light: Light source (uses default if None)
            material: Material properties (uses default if None)
            shininess: Shininess exponent for specular highlights
            view_matrix: View matrix for transforming light to view space
        """
        light = light or self.DEFAULT_LIGHT
        material = material or self.DEFAULT_MATERIAL
        
        # Create I_light matrix: [diffuse, specular, ambient]
        I_light = np.array([
            light.diffuse,
            light.specular,
            light.ambient
        ], dtype=np.float32)
        
        # Create K_materials matrix: [diffuse, specular, ambient]
        K_materials = np.array([
            material.diffuse,
            material.specular,
            material.ambient
        ], dtype=np.float32)
        
        # Upload uniforms (these will be used in vertex shader)
        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light.position, 'light_pos_world')
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        
        # Upload view matrix for light transform
        if view_matrix is not None:
            self.uma.upload_uniform_matrix4fv(view_matrix, 'view', True)
