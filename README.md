# OpenEPA Analysis Code

This repository contains analysis code for the OpenEPA project.

## Overview

OpenEPA is a tool for environmental protection analysis and data processing.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/yourusername/OpenEPA.git
cd OpenEPA
pip install -r requirements.txt
```

## Usage

Describe how to use your code here.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/OpenEPA](https://github.com/yourusername/OpenEPA)

## What is OpenEPA?

OpenEPA is an open-source tool designed to simplify access to the Climate and Economic Justice Screening Tool (CEJST) dataset. It enables users to:

- Query CEJST data using natural language
- Get AI-driven analyses of environmental justice data
- Access demographic insights for specific locations
- Filter data by State, County, or Census Tract ID

### The Problem

The CEJST dataset, while valuable, can be challenging to navigate without specialized tools or expertise. This creates barriers for:

- Community groups
- Grant writers
- Local planners
- Concerned citizens

### The Solution

OpenEPA features:

- User-friendly web interface
- Natural language query processing
- AI-powered data summaries
- Filtered location-based analysis

### Technical Details

The application uses:

- Python Flask backend
- Pandas for data processing
- OpenAI API for query analysis
- Standard web technologies (HTML, CSS, JS)

For detailed technical documentation and setup instructions, see our [documentation](docs/README.md).

## Core Dataset

OpenEPA utilizes CEJST version 2.0, which includes:

- Disadvantaged community identifiers
- Tribal lands information
- Multiple environmental burden categories
- Climate change impacts
- Socioeconomic indicators
