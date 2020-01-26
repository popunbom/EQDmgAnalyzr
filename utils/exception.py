# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29
from textwrap import dedent


class CannotFindVariable( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


class InvalidImageOrFile( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


class UnsupportedDataType( Exception ):
    
    
    def __init__( self, detail: str, **kwargs ) -> None:
        message = "Data Type is unsupported: ({detail})".format(
            detail=detail
        )
        super().__init__( message )
        self.kwargs = kwargs

class UnsupportedOption( Exception ):

    def __init__(self, detail: str, available_options=None, **kwargs) -> None:
        message = dedent("""
        Option is not supported ({detail})
          Available Options: {options}
        """)
        message = "".format(
            detail=detail,
            options="None" if available_options is None else ", ".join(
                [str(x) for x in available_options]
            )
        )
        super().__init__( message )
        self.kwargs = kwargs
