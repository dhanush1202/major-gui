import * as React from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

export default function LoadingComponent() {
  return (
    <div className='loading-overlay'>
      <Box sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(255, 255, 255, 0.8)', // Adjust transparency and color here
        zIndex: 9999, // Set a high z-index to ensure it's above other elements
      }}>
        <CircularProgress />
      </Box>
    </div>
  );
}
