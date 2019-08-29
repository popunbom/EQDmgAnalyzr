# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29


class CannotFindVariable( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


class InvalidImageOrFile( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


class UnsupportedDataTypes( Exception ):
    
    
    def __init__( self, detail: str, **kwargs ) -> None:
        message = "Unsupported Data Types ({detail})".format(
            detail=detail
        )
        super().__init__( message )
        self.kwargs = kwargs
